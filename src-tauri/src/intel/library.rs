use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

#[cfg(not(test))]
const BUILTIN_LIBRARY_JSON: &str = include_str!("../../resources/intel/mitre_core.v1.json");

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LibraryFile {
    pub schema_version: u32,
    pub library_id: String,
    pub techniques: Vec<Technique>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Technique {
    pub technique_id: String,
    pub name: String,
    pub tactics: Vec<Tactic>,
    pub aliases: Vec<String>,
    pub keywords: Vec<Keyword>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tactic {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Keyword {
    pub id: String,
    pub pattern: String,
    #[serde(rename = "match")]
    pub match_kind: MatchKind,
    pub columns: Vec<String>,
    pub score: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    Substring,
    Word,
}

#[derive(Debug, Clone)]
pub struct LoadedLibrary {
    pub library_ids: Vec<String>,
    pub techniques: Vec<Technique>,
    pub library_hash: String,
    pub custom_library_error: Option<String>,
}

impl LoadedLibrary {
    pub fn technique_count(&self) -> usize {
        self.techniques.len()
    }

    pub fn keyword_count(&self) -> usize {
        self.techniques.iter().map(|t| t.keywords.len()).sum()
    }
}

pub fn load_builtin_library() -> Result<LoadedLibrary> {
    let builtin_raw = builtin_library_json()?;
    let builtin = parse_library("built-in MITRE core", builtin_raw.as_ref())?;
    let library_hash = hash_library_sources(&[builtin_raw.as_ref()]);
    Ok(LoadedLibrary {
        library_ids: vec![builtin.library_id],
        techniques: builtin.techniques,
        library_hash,
        custom_library_error: None,
    })
}

pub fn load_merged_library() -> Result<LoadedLibrary> {
    let builtin_raw = builtin_library_json()?;
    let builtin = parse_library("built-in MITRE core", builtin_raw.as_ref())?;
    let mut library_ids = vec![builtin.library_id];
    let mut techniques = builtin.techniques;
    let mut hash_sources = vec![builtin_raw.into_owned()];
    let mut custom_library_error = None;

    let custom_path = custom_library_path();
    if custom_path.is_file() {
        match std::fs::read_to_string(&custom_path)
            .with_context(|| format!("reading {}", custom_path.display()))
            .and_then(|raw| {
                let custom =
                    parse_library(&format!("custom library {}", custom_path.display()), &raw)?;
                Ok((raw, custom))
            }) {
            Ok((raw, custom)) => {
                library_ids.push(custom.library_id);
                techniques.extend(custom.techniques);
                hash_sources.push(raw);
            }
            Err(err) => {
                let message = format!(
                    "Ignoring invalid custom intel library {}: {err}",
                    custom_path.display()
                );
                eprintln!("{message}");
                custom_library_error = Some(message);
            }
        }
    }

    let hash_refs: Vec<&str> = hash_sources.iter().map(String::as_str).collect();
    Ok(LoadedLibrary {
        library_ids,
        techniques,
        library_hash: hash_library_sources(&hash_refs),
        custom_library_error,
    })
}

#[cfg(not(test))]
fn builtin_library_json() -> Result<Cow<'static, str>> {
    Ok(Cow::Borrowed(BUILTIN_LIBRARY_JSON))
}

#[cfg(test)]
fn builtin_library_json() -> Result<Cow<'static, str>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("intel")
        .join("mitre_core.v1.json");
    std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))
        .map(Cow::Owned)
}

pub fn custom_library_path() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("log-parser")
        .join("intel")
        .join("custom_library.v1.json")
}

fn parse_library(label: &str, raw: &str) -> Result<LibraryFile> {
    let library: LibraryFile =
        serde_json::from_str(raw).with_context(|| format!("parsing {label}"))?;
    validate_library(label, &library)?;
    Ok(library)
}

fn validate_library(label: &str, library: &LibraryFile) -> Result<()> {
    if library.schema_version != 1 {
        bail!(
            "{label} has unsupported schemaVersion {}",
            library.schema_version
        );
    }
    if library.library_id.trim().is_empty() {
        bail!("{label} has an empty libraryId");
    }
    if library.techniques.is_empty() {
        bail!("{label} has no techniques");
    }

    for technique in &library.techniques {
        let technique_label = if technique.technique_id.trim().is_empty() {
            "<empty techniqueId>"
        } else {
            technique.technique_id.as_str()
        };
        if technique.technique_id.trim().is_empty() {
            bail!("{label} has a technique with an empty techniqueId");
        }
        if technique.name.trim().is_empty() {
            bail!("{label} technique {technique_label} has an empty name");
        }
        if technique.tactics.is_empty() {
            bail!("{label} technique {technique_label} has no tactics");
        }
        if technique.keywords.is_empty() {
            bail!("{label} technique {technique_label} has no keywords");
        }
        for tactic in &technique.tactics {
            if tactic.id.trim().is_empty() || tactic.name.trim().is_empty() {
                bail!("{label} technique {technique_label} has an empty tactic id/name");
            }
        }
        for keyword in &technique.keywords {
            if keyword.id.trim().is_empty() {
                bail!("{label} technique {technique_label} has a keyword with an empty id");
            }
            if keyword.pattern.trim().is_empty() {
                bail!("{label} keyword {} has an empty pattern", keyword.id);
            }
            if !(0..=100).contains(&keyword.score) {
                bail!(
                    "{label} keyword {} has score {} outside 0..=100",
                    keyword.id,
                    keyword.score
                );
            }
        }
    }

    if library
        .techniques
        .iter()
        .flat_map(|t| t.keywords.iter())
        .next()
        .is_none()
    {
        return Err(anyhow!("{label} has no keyword indicators"));
    }

    Ok(())
}

fn hash_library_sources(sources: &[&str]) -> String {
    let mut hasher = DefaultHasher::new();
    for source in sources {
        source.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_library_loads_real_resource() {
        let library = load_builtin_library().unwrap();

        assert_eq!(library.library_ids, vec!["mitre_core_v1"]);
        // Sanity floors, not exact counts - the built-in library is expected to keep growing as
        // more MITRE ATT&CK coverage gets added. This just guards against a genuinely broken/
        // near-empty load, not against legitimate expansion.
        assert!(
            library.technique_count() >= 60,
            "expected at least 60 techniques, got {}",
            library.technique_count()
        );
        assert!(
            library.keyword_count() >= 300,
            "expected at least 300 keywords, got {}",
            library.keyword_count()
        );
        assert!(!library.library_hash.is_empty());
        assert!(library.custom_library_error.is_none());
    }
}
