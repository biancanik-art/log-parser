use crate::db::ColumnMeta;
use std::collections::HashSet;

/// Turns raw header cells into SQL-safe, deduplicated, non-empty identifiers. `row_num` is
/// reserved for the synthetic primary key so a source column literally named "row_num" won't
/// collide with it.
pub fn sanitize_headers(raw: &[String]) -> Vec<ColumnMeta> {
    let mut used: HashSet<String> = HashSet::new();
    used.insert("row_num".to_string());

    raw.iter()
        .enumerate()
        .map(|(idx, original)| {
            let mut base = sanitize_one(original);
            if base.is_empty() {
                base = format!("column_{idx}");
            }
            if base.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                base = format!("c_{base}");
            }

            let mut candidate = base.clone();
            let mut n = 2;
            while used.contains(&candidate) {
                candidate = format!("{base}_{n}");
                n += 1;
            }
            used.insert(candidate.clone());

            ColumnMeta {
                sql_name: candidate,
                original_name: if original.trim().is_empty() {
                    format!("Column {}", idx + 1)
                } else {
                    original.clone()
                },
                col_index: idx,
                inferred_type: infer_type(original),
            }
        })
        .collect()
}

fn sanitize_one(header: &str) -> String {
    let mut out = String::new();
    let mut last_was_underscore = false;
    for c in header.chars() {
        let lower = c.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            out.push(lower);
            last_was_underscore = false;
        } else if !last_was_underscore {
            out.push('_');
            last_was_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// Header-name-only heuristic used solely to pick sensible default filter operators in the UI.
/// Never enforced at the storage layer and never used to normalize/rename columns across sources.
fn infer_type(header: &str) -> String {
    let h = header.to_ascii_lowercase();
    if h.contains("time")
        || h.contains("date")
        || h.contains("generated")
        || h.contains("createdat")
    {
        "timestamp"
    } else if h.contains("ip") || h.contains("address") {
        "ip"
    } else if h.contains("id") || h.contains("guid") {
        "identifier"
    } else {
        "text"
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_headers_dedupes_and_handles_edge_cases() {
        let raw = vec![
            "TimeGenerated".to_string(),
            "".to_string(),
            "Account".to_string(),
            "Account".to_string(),
            "1stThing".to_string(),
            "row_num".to_string(),
            "Weird!! Header--Name".to_string(),
        ];
        let cols = sanitize_headers(&raw);
        let names: Vec<&str> = cols.iter().map(|c| c.sql_name.as_str()).collect();

        assert_eq!(names[0], "timegenerated");
        assert_eq!(names[1], "column_1");
        assert_eq!(names[2], "account");
        assert_eq!(names[3], "account_2");
        assert_eq!(names[4], "c_1stthing");
        assert_eq!(names[5], "row_num_2"); // collides with reserved row_num
        assert_eq!(names[6], "weird_header_name");

        let unique: HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn infer_type_heuristics() {
        assert_eq!(infer_type("TimeGenerated"), "timestamp");
        assert_eq!(infer_type("EventDate"), "timestamp");
        assert_eq!(infer_type("SrcIpAddress"), "ip");
        assert_eq!(infer_type("EventID"), "identifier");
        assert_eq!(infer_type("CommandLine"), "text");
    }
}
