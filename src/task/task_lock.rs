//! Locks a task holds while its commands run, so runs that need the same resource take turns.
//!
//! `lock = "name"` names a resource on this machine, shared by every task that names it;
//! `lock = true` locks the task itself across every worktree of its repository. Each lock is
//! a file under `$MISE_STATE_DIR/task-locks/` held under an OS file lock (`flock`, or
//! `LockFileEx` on Windows) for as long as the task runs, so the kernel releases it when the
//! holder dies and nothing ever goes stale. Lock files are never deleted: a waiter that had
//! opened a removed file would lock it while a newcomer locked its replacement.
//!
//! `lock = { name, slots = N }` lets up to N runs hold a named lock at once: it has one file
//! for each slot, and a run takes whichever of its first N is free. The first slot's file is
//! the one a plain `lock = "name"` takes, so a run without slots only ever shares the first.
//!
//! The holder writes who it is beside the slot it holds, for the message a waiter prints. A
//! record whose slot can be taken is left over and is never shown.
//!
//! Locks are taken in sorted order, so two runs never each hold a lock the other wants. A task
//! exports what it holds in `MISE_TASK_LOCKS`, and a task that inherits a lock there, from a
//! nested `mise run` or from a `run = [{ task }]` entry, treats it as already held instead of
//! waiting on its own parent.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eyre::{Result, WrapErr, bail};
use serde::{Deserialize, Serialize};

use crate::dirs;
use crate::file::display_path;
use crate::task::{NamedLock, Task, TaskLock};

/// The locks a task's commands run under, comma-separated.
pub(crate) const LOCKS_ENV: &str = "MISE_TASK_LOCKS";

/// How often a waiter says it is still waiting.
const REPORT_EVERY: Duration = Duration::from_secs(30);
const FIRST_POLL: Duration = Duration::from_millis(100);
const MAX_POLL: Duration = Duration::from_secs(1);

/// One lock a task asks for.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LockKey {
    /// What `MISE_TASK_LOCKS` carries and the lock file is named after.
    id: String,
    /// What messages call it.
    display: String,
    /// How many runs may hold it at once.
    slots: NonZeroU32,
}

impl LockKey {
    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    fn named(lock: &NamedLock) -> Self {
        Self {
            id: lock.name.clone(),
            display: lock.name.clone(),
            slots: lock.slots,
        }
    }

    /// `lock = true`: the task's name in its repository. Every worktree of a repository shares
    /// one git common dir, so they share this key.
    fn this_task(task: &Task) -> Self {
        let dir = task
            .config_root
            .clone()
            .or_else(|| task.config_source.parent().map(Path::to_path_buf))
            .unwrap_or_default();
        let repo = repository_of(&dir);
        let id = format!(
            "task:{}",
            &mise_util::hash::hash_sha256_to_str(&format!(
                "{}\0{}",
                repo.to_string_lossy(),
                task.name
            ))[..16]
        );
        Self {
            id,
            display: format!("{} (this repository)", task.name),
            slots: NonZeroU32::MIN,
        }
    }

    /// The file for one of the lock's slots. The first slot keeps the name a lock without
    /// slots has, so the two share it.
    fn path(&self, slot: u32, ext: &str) -> PathBuf {
        let file = &mise_util::hash::hash_sha256_to_str(&self.id)[..16];
        match slot {
            0 => locks_dir().join(format!("{file}.{ext}")),
            _ => locks_dir().join(format!("{file}.{slot}.{ext}")),
        }
    }

    /// The lock's name, and how many runs may hold it when that is more than one.
    fn described(&self) -> String {
        match self.slots.get() {
            1 => self.display.clone(),
            n => format!("{} ({n} slots)", self.display),
        }
    }
}

fn locks_dir() -> PathBuf {
    dirs::STATE.join("task-locks")
}

/// The git directory every worktree of the repository holding `dir` shares, or `dir` itself
/// outside git.
fn repository_of(dir: &Path) -> PathBuf {
    let dir = dunce::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    for root in dir.ancestors() {
        let dot_git = root.join(".git");
        if dot_git.is_dir() {
            return dunce::canonicalize(&dot_git).unwrap_or(dot_git);
        }
        // A linked worktree or a submodule has a file naming its private git dir; a linked
        // worktree's private dir names the shared one in `commondir`.
        if let Ok(text) = std::fs::read_to_string(&dot_git)
            && let Some(target) = text.trim().strip_prefix("gitdir:")
        {
            let git_dir = root.join(target.trim());
            let shared = match std::fs::read_to_string(git_dir.join("commondir")) {
                Ok(common) => git_dir.join(common.trim()),
                Err(_) => git_dir,
            };
            return dunce::canonicalize(&shared).unwrap_or(shared);
        }
    }
    dir
}

/// The locks `task` asks for, sorted and without repeats.
pub(crate) fn requested(task: &Task) -> Vec<LockKey> {
    let mut keys = match &task.lock {
        None | Some(TaskLock::Off) => vec![],
        Some(TaskLock::Task) => vec![LockKey::this_task(task)],
        Some(TaskLock::Names(names)) => names.iter().map(LockKey::named).collect(),
    };
    keys.sort();
    // A name listed twice is one lock; sorting put the entry with the fewest slots first.
    keys.dedup_by(|later, earlier| later.id == earlier.id);
    keys
}

/// How `mise tasks info` and `--dry-run` describe a task's locks.
pub fn describe(task: &Task) -> Option<String> {
    let keys = requested(task);
    (!keys.is_empty()).then(|| {
        keys.iter()
            .map(LockKey::described)
            .collect::<Vec<_>>()
            .join(", ")
    })
}

/// The locks a parent task already holds for this one.
pub(crate) fn inherited(env: &BTreeMap<String, String>) -> BTreeSet<String> {
    env.get(LOCKS_ENV)
        .map(|value| {
            value
                .split(',')
                .filter(|id| !id.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Every run takes locks in one order, sorted by id, so no two runs can each hold a lock the
/// other waits for. A task started under a locked task already holds what it inherited, so it
/// may only add locks that sort after all of them: one parent holding `a` whose child wants `b`
/// and another holding `b` whose child wants `a` would otherwise wait for each other forever.
pub(crate) fn check_nested_order(
    task: &Task,
    inherited: &BTreeSet<String>,
    to_take: &[LockKey],
) -> Result<()> {
    let (Some(held), Some(first)) = (inherited.last(), to_take.first()) else {
        return Ok(());
    };
    if first.id.as_str() < held.as_str() {
        bail!(
            "task {}: cannot take lock {} while a parent task holds {held}: a nested task can only \
             add locks that sort after the ones it inherits. Add {} to the parent task's `lock`",
            task.name,
            first.display,
            first.display
        );
    }
    Ok(())
}

/// What `MISE_TASK_LOCKS` says once this task holds `keys` on top of what it inherited.
pub(crate) fn exported(inherited: &BTreeSet<String>, keys: &[LockKey]) -> String {
    let mut all = inherited.clone();
    all.extend(keys.iter().map(|k| k.id.clone()));
    all.into_iter().collect::<Vec<_>>().join(",")
}

/// A wait or a holder's age: tenths of a second under a minute, whole units above.
fn human(d: Duration) -> String {
    let secs = d.as_secs();
    match secs {
        0..60 => format!("{:.1}s", d.as_secs_f64()),
        60..3600 => format!("{}m{:02}s", secs / 60, secs % 60),
        _ => format!("{}h{:02}m", secs / 3600, secs % 3600 / 60),
    }
}

/// Who holds a lock, as the holder wrote it.
#[derive(Debug, Serialize, Deserialize)]
struct HolderRecord {
    pid: u32,
    task: String,
    root: Option<PathBuf>,
    started_at: u64,
}

impl HolderRecord {
    fn read(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn describe(&self) -> String {
        let since = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .saturating_sub(Duration::from_secs(self.started_at));
        let since = human(since);
        match &self.root {
            Some(root) => format!(
                "{} in {} (pid {}, for {since})",
                self.task,
                display_path(root),
                self.pid
            ),
            None => format!("{} (pid {}, for {since})", self.task, self.pid),
        }
    }
}

/// One slot of a lock, held.
struct HeldSlot {
    key: LockKey,
    slot: u32,
    _file: fslock::LockFile,
}

/// Locks this task holds. Dropping it removes the holder records, then releases the locks.
pub(crate) struct HeldLocks {
    locks: Vec<HeldSlot>,
}

impl HeldLocks {
    /// Whether this task took no lock of its own, because it inherited every one it names.
    pub(crate) fn is_empty(&self) -> bool {
        self.locks.is_empty()
    }
}

impl Drop for HeldLocks {
    fn drop(&mut self) {
        // The record goes first, while the lock still keeps the next holder from writing its own.
        for held in &self.locks {
            let _ = std::fs::remove_file(held.key.path(held.slot, "json"));
        }
    }
}

/// Taking a task's locks, one at a time in sorted order.
pub(crate) struct LockAcquisition<'a> {
    task: &'a Task,
    pending: std::vec::IntoIter<LockKey>,
    /// The lock this acquisition is waiting for, each of its slots already opened.
    current: Option<(LockKey, Vec<fslock::LockFile>)>,
    held: HeldLocks,
}

impl<'a> LockAcquisition<'a> {
    pub(crate) fn new(task: &'a Task, keys: Vec<LockKey>) -> Result<Self> {
        let dir = locks_dir();
        std::fs::create_dir_all(&dir)
            .wrap_err_with(|| format!("task {}: cannot create {}", task.name, dir.display()))?;
        Ok(Self {
            task,
            pending: keys.into_iter(),
            current: None,
            held: HeldLocks { locks: vec![] },
        })
    }

    /// Takes every lock that is free, in order, and stops at the first one that is not.
    /// A lock is free when any of its slots is. `true` once every lock is held.
    pub(crate) fn try_take(&mut self) -> Result<bool> {
        loop {
            let (key, mut files) = match self.current.take() {
                Some(current) => current,
                None => match self.pending.next() {
                    Some(key) => {
                        let files = (0..key.slots.get())
                            .map(|slot| {
                                let path = key.path(slot, "lock");
                                fslock::LockFile::open(&path).wrap_err_with(|| {
                                    format!(
                                        "task {}: cannot open lock file {}",
                                        self.task.name,
                                        path.display()
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        (key, files)
                    }
                    None => return Ok(true),
                },
            };
            let mut taken = None;
            for (slot, file) in files.iter_mut().enumerate() {
                if file.try_lock().wrap_err_with(|| {
                    format!("task {}: cannot take lock {}", self.task.name, key.display)
                })? {
                    taken = Some(slot);
                    break;
                }
            }
            let Some(slot) = taken else {
                self.current = Some((key, files));
                return Ok(false);
            };
            // Closing the other slots' files keeps nothing: this run never locked them.
            let file = files.swap_remove(slot);
            let slot = slot as u32;
            self.write_record(&key, slot);
            self.held.locks.push(HeldSlot {
                key,
                slot,
                _file: file,
            });
        }
    }

    /// The lock this acquisition waits for and who holds it, as far as the holders said.
    pub(crate) fn waiting_for(&self) -> String {
        let Some((key, _)) = &self.current else {
            return String::new();
        };
        let holders = (0..key.slots.get())
            .filter_map(|slot| HolderRecord::read(&key.path(slot, "json")))
            .map(|record| record.describe())
            .collect::<Vec<_>>();
        let holders = match holders.is_empty() {
            true => String::new(),
            false => format!(", held by {}", holders.join(", ")),
        };
        format!("lock {}{holders}", key.described())
    }

    /// Waits until every lock is held. `waited` is the time this task already spent waiting
    /// for locks, and this wait adds to it: a task that gave its locks back to wait for a
    /// `--jobs` slot keeps only the `timeout` it has left, and the slot wait does not count.
    /// `report` prints a line now and every 30 seconds; `check_interruption` ends the wait as
    /// soon as the run is stopping.
    pub(crate) async fn wait(
        mut self,
        timeout: Option<Duration>,
        waited: &mut Duration,
        report: impl Fn(&str),
        check_interruption: impl Fn() -> Result<()>,
    ) -> Result<HeldLocks> {
        let start = Instant::now();
        let mut reported_at: Option<Instant> = None;
        let mut poll = FIRST_POLL;
        while !self.try_take()? {
            check_interruption()?;
            if let Some(timeout) = timeout
                && *waited + start.elapsed() >= timeout
            {
                bail!(
                    "task {}: timed out after {} waiting for {}",
                    self.task.name,
                    human(timeout),
                    self.waiting_for()
                );
            }
            if reported_at.is_none_or(|at| at.elapsed() >= REPORT_EVERY) {
                report(&format!("waiting for {}", self.waiting_for()));
                reported_at = Some(Instant::now());
            }
            tokio::time::sleep(poll).await;
            poll = (poll * 2).min(MAX_POLL);
        }
        if reported_at.is_some() {
            report(&format!(
                "got {} after {}",
                self.held_names(),
                human(*waited + start.elapsed())
            ));
        }
        *waited += start.elapsed();
        Ok(self.held)
    }

    /// Every lock, once `try_take` returned `true`.
    pub(crate) fn into_held(self) -> HeldLocks {
        debug_assert!(self.current.is_none());
        self.held
    }

    fn held_names(&self) -> String {
        let names = self
            .held
            .locks
            .iter()
            .map(|held| held.key.display.as_str())
            .collect::<Vec<_>>();
        format!(
            "{} {}",
            if names.len() == 1 { "lock" } else { "locks" },
            names.join(", ")
        )
    }

    fn write_record(&self, key: &LockKey, slot: u32) {
        let record = HolderRecord {
            pid: std::process::id(),
            task: self.task.name.clone(),
            root: self.task.config_root.clone(),
            started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default(),
        };
        // Only for the message other runs print, so a failure here changes nothing else.
        if let Ok(json) = serde_json::to_string(&record)
            && let Err(err) = crate::file::write(key.path(slot, "json"), json)
        {
            debug!("task {}: cannot record lock holder: {err}", self.task.name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(name: &str) -> LockKey {
        LockKey::named(&NamedLock::new(name))
    }

    fn slotted(name: &str, slots: u32) -> LockKey {
        LockKey::named(&NamedLock::new(name).with_slots(NonZeroU32::new(slots).unwrap()))
    }

    fn task(name: &str) -> Task {
        Task {
            name: name.into(),
            ..Default::default()
        }
    }

    fn worktree(main: &Path, name: &str) -> PathBuf {
        let private = main.join(".git").join("worktrees").join(name);
        std::fs::create_dir_all(&private).unwrap();
        std::fs::write(private.join("commondir"), "../..\n").unwrap();
        let root = main.join(name);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join(".git"),
            format!("gitdir: {}\n", private.display()),
        )
        .unwrap();
        root
    }

    #[test]
    fn worktrees_share_their_repository() {
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("repo");
        std::fs::create_dir_all(main.join(".git")).unwrap();
        let linked = worktree(&main, "feature");
        let nested = main.join("crates").join("a");
        std::fs::create_dir_all(&nested).unwrap();

        let shared = repository_of(&main);
        assert_eq!(repository_of(&linked), shared);
        assert_eq!(repository_of(&nested), shared);

        let other = tmp.path().join("other");
        std::fs::create_dir_all(other.join(".git")).unwrap();
        assert_ne!(repository_of(&other), shared);
    }

    #[test]
    fn nested_tasks_only_add_locks_after_the_ones_they_inherit() {
        let task = Task {
            name: "child".into(),
            ..Default::default()
        };
        let held: BTreeSet<String> = ["b".to_string()].into();
        assert!(check_nested_order(&task, &held, &[named("c")]).is_ok());
        assert!(check_nested_order(&task, &BTreeSet::new(), &[named("a")]).is_ok());
        assert!(check_nested_order(&task, &held, &[]).is_ok());
        let err = check_nested_order(&task, &held, &[named("a")]).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot take lock a while a parent task holds b"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_retry_keeps_the_time_left_to_wait() {
        let task = Task {
            name: "retry".into(),
            ..Default::default()
        };
        let key = named(&format!("budget-{}", std::process::id()));
        std::fs::create_dir_all(locks_dir()).unwrap();
        let mut holder = fslock::LockFile::open(&key.path(0, "lock")).unwrap();
        assert!(holder.try_lock().unwrap());

        let mut acquisition = LockAcquisition::new(&task, vec![key]).unwrap();
        assert!(!acquisition.try_take().unwrap());
        // Five seconds already spent waiting before the locks were given back.
        let mut waited = Duration::from_secs(5);
        let started = Instant::now();
        let err = acquisition
            .wait(Some(Duration::from_secs(1)), &mut waited, |_| {}, || Ok(()))
            .await
            .err()
            .expect("the budget was already spent");
        assert!(err.to_string().contains("timed out after 1.0s"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(1));

        // Once the lock is free, the wait adds its own time to what the task spent before.
        drop(holder);
        let key = named(&format!("budget-{}", std::process::id()));
        let acquisition = LockAcquisition::new(&task, vec![key]).unwrap();
        let mut waited = Duration::from_secs(2);
        let held = acquisition
            .wait(
                Some(Duration::from_secs(10)),
                &mut waited,
                |_| {},
                || Ok(()),
            )
            .await
            .unwrap();
        drop(held);
        assert!(waited >= Duration::from_secs(2) && waited < Duration::from_secs(3));
    }

    #[test]
    fn durations_read_at_a_glance() {
        assert_eq!(human(Duration::from_millis(1500)), "1.5s");
        assert_eq!(human(Duration::from_secs(182)), "3m02s");
        assert_eq!(human(Duration::from_secs(7260)), "2h01m");
    }

    #[test]
    fn inherited_locks_round_trip_through_the_env() {
        let keys = vec![named("b"), named("a")];
        let parent: BTreeSet<String> = ["c".to_string()].into();
        let value = exported(&parent, &keys);
        assert_eq!(value, "a,b,c");
        let env = BTreeMap::from([(LOCKS_ENV.to_string(), value)]);
        assert_eq!(
            inherited(&env),
            ["a", "b", "c"].map(String::from).into_iter().collect()
        );
        assert!(inherited(&BTreeMap::new()).is_empty());
    }

    #[test]
    fn a_lock_with_slots_admits_that_many_runs() {
        let name = format!("slots-{}", std::process::id());
        let (a, b, c) = (task("a"), task("b"), task("c"));

        let mut first = LockAcquisition::new(&a, vec![slotted(&name, 2)]).unwrap();
        assert!(first.try_take().unwrap());
        let first = first.into_held();
        let mut second = LockAcquisition::new(&b, vec![slotted(&name, 2)]).unwrap();
        assert!(second.try_take().unwrap());
        let second = second.into_held();

        let mut third = LockAcquisition::new(&c, vec![slotted(&name, 2)]).unwrap();
        assert!(!third.try_take().unwrap());
        let waiting = third.waiting_for();
        assert!(
            waiting.starts_with(&format!("lock {name} (2 slots), held by a (pid ")),
            "{waiting}"
        );
        assert!(waiting.contains(", b (pid "), "{waiting}");

        // Whichever run ends, its slot is the one the waiter takes.
        drop(first);
        assert!(third.try_take().unwrap());
        drop(third.into_held());
        drop(second);
    }

    #[test]
    fn a_lock_without_slots_shares_the_first_slot() {
        let name = format!("first-slot-{}", std::process::id());
        let (a, b) = (task("a"), task("b"));

        let mut slotted_run = LockAcquisition::new(&a, vec![slotted(&name, 2)]).unwrap();
        assert!(slotted_run.try_take().unwrap());
        let slotted_run = slotted_run.into_held();
        let mut plain = LockAcquisition::new(&b, vec![named(&name)]).unwrap();
        assert!(!plain.try_take().unwrap());
        assert!(
            plain
                .waiting_for()
                .starts_with(&format!("lock {name}, held by a"))
        );
        drop(slotted_run);
        assert!(plain.try_take().unwrap());
        drop(plain.into_held());
    }

    #[test]
    fn a_name_listed_twice_is_one_lock() {
        let mut t = task("t");
        t.lock = Some(TaskLock::Names(vec![
            NamedLock::new("xcode").with_slots(NonZeroU32::new(3).unwrap()),
            NamedLock::new("sim"),
            NamedLock::new("xcode").with_slots(NonZeroU32::new(2).unwrap()),
        ]));
        assert_eq!(requested(&t), vec![named("sim"), slotted("xcode", 2)]);
        assert_eq!(describe(&t).unwrap(), "sim, xcode (2 slots)");
    }
}
