use rayon::{join, prelude::*};
use regex::{Regex, RegexBuilder};
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (dictionary, stopwords) = join(build_dictionary, build_stopwords);

    let files = glob::glob("./_wordpress_posts/**/*.md")
        .expect("Failed to read _wordpress_posts directory");

    let analyzed_files: Vec<(String, HashMap<String, u32>)> = files
        .par_bridge()
        .map(|path| {
            path.ok()
                .and_then(|x| x.to_str().map(String::from))
                .and_then(|x| analyze_path(x.as_str(), &dictionary, &stopwords).ok())
                .unwrap()
        })
        .collect();

    let similarities_per_file = calculate_all_similarities(&analyzed_files);

    const TOP_ITEMS: usize = 3;

    let top_similarities_per_files = similarities_per_file
        .iter()
        .collect::<Vec<(&String, &HashMap<String, f32>)>>()
        .par_iter()
        .map(|(permalink, val)| {
            let mut entries = val.iter().collect::<Vec<(&String, &f32)>>();
            entries.par_sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

            let top_permalinks = entries[..TOP_ITEMS]
                .to_vec()
                .into_iter()
                .map(|(key, _)| key)
                .collect::<Vec<&String>>();
            (*permalink, top_permalinks)
        })
        .collect::<HashMap<&String, Vec<&String>>>();

    let json = serde_json::to_string(&top_similarities_per_files)?;
    std::fs::write("./results.json", json).expect("Couldn't write results.json");

    Ok(())
}

fn calculate_all_similarities(
    r: &Vec<(String, HashMap<String, u32>)>,
) -> HashMap<String, HashMap<String, f32>> {
    let mut all_results: HashMap<String, HashMap<String, f32>> = HashMap::new();
    for (permalink, _) in r.to_owned() {
        all_results.insert(permalink, HashMap::new());
    }
    for (permalink1, counter1) in r.iter() {
        for (permalink2, counter2) in r.iter() {
            if permalink1 == permalink2 {
                continue;
            }
            let value: f32 = *all_results
                .get_mut(permalink1)
                .unwrap()
                .entry(permalink2.to_owned())
                .or_insert(calculate_cosine_similarity(&counter1, &counter2));

            all_results
                .get_mut(permalink1)
                .unwrap()
                .insert(permalink2.to_owned(), value);
            all_results
                .get_mut(permalink2)
                .unwrap()
                .insert(permalink1.to_owned(), value);
        }
    }
    all_results
}

fn calculate_cosine_similarity(
    counter1: &HashMap<String, u32>,
    counter2: &HashMap<String, u32>,
) -> f32 {
    let k1: HashSet<String> = counter1.keys().cloned().collect();
    let k2: HashSet<String> = counter2.keys().cloned().collect();
    let common_keys: HashSet<String> = k1.intersection(&k2).cloned().collect();

    let r1: u32 = counter1.values().map(|x| x * x).sum();
    let r2: u32 = counter2.values().map(|x| x * x).sum();

    let sum: u32 = common_keys
        .into_par_iter()
        .map(|key| counter1.get(&key).unwrap_or(&1) * counter2.get(&key).unwrap_or(&1))
        .sum();
    (sum as f32 / (r1 as f32).sqrt() / (r2 as f32).sqrt()).clamp(0.0, 1.0)
}

fn analyze_path(
    path: &str,
    dictionary: &HashMap<String, String>,
    stopwords: &HashSet<String>,
) -> Result<(String, HashMap<String, u32>), Box<dyn std::error::Error + 'static>> {
    let article = String::from_utf8(std::fs::read(path)?)?.to_lowercase();
    let permalink = get_permalink(&article);

    let article = clean_up(&article);

    let counter = count_words(&article, &dictionary, &stopwords);

    Ok((permalink, counter))
}

fn count_words(
    article: &String,
    dictionary: &HashMap<String, String>,
    stopwords: &HashSet<String>,
) -> HashMap<String, u32> {
    let mut counter: HashMap<String, u32> = HashMap::new();
    let words: Vec<String> = article
        .split_whitespace()
        .filter_map(|word| -> Option<String> {
            let w = word.trim();
            if w.len() > 1 && !w.starts_with('\\') && !stopwords.contains(w) {
                Some(dictionary.get(w).map_or_else(
                    || {
                        println!("Missing dict for word {}", w);
                        w.to_string()
                    },
                    |w| w.to_owned(),
                ))
            } else {
                None
            }
        })
        .collect();

    for word in words {
        *counter.entry(word.to_string()).or_insert(0) += 1;
    }

    counter
}

fn get_permalink(article: &String) -> String {
    let permalink_pattern = Regex::new(r"permalink:\s*(.*)").unwrap();
    let permalink = permalink_pattern
        .captures(article)
        .unwrap()
        .get(1)
        .map(|m| m.as_str())
        .unwrap();
    permalink.to_string()
}

fn clean_up(article: &String) -> String {
    let article = article.split("---").collect::<Vec<&str>>();
    let article = article.get(2).unwrap().to_string();

    let code_pattern = RegexBuilder::new(r"```\w{2,7}?.*?```")
        .case_insensitive(true)
        .dot_matches_new_line(true)
        .build()
        .unwrap();
    let inline_code_pattern = Regex::new(r"`[^`]*?`").unwrap();
    let link_pattern = Regex::new(r"\[(.*?)\]\(.*?\)").unwrap();
    let html_pattern = Regex::new(r"<[^>]*?>").unwrap();
    let punctuation_pattern = Regex::new(r"[-–—_,;:!?.'”„()\[\]{}/#@$%^&*<>|`=>]").unwrap();
    let article = code_pattern.replace_all(&article, " ").to_string();
    let article = inline_code_pattern.replace_all(&article, " ").to_string();
    let article = link_pattern.replace_all(&article, "$1").to_string();
    let article = html_pattern.replace_all(&article, " ").to_string();
    let article = punctuation_pattern.replace_all(&article, " ").to_string();
    article
}

fn build_stopwords() -> HashSet<String> {
    println!("Reading stopwords file…");
    let file = std::fs::File::open("./stopwords.txt").unwrap();
    let dict = std::io::BufReader::new(file)
        .lines()
        .par_bridge()
        .map(|x| x.unwrap().trim().to_lowercase());
    println!("Building stopwords Set…");
    HashSet::from_par_iter(dict)
}

fn build_dictionary() -> HashMap<String, String> {
    println!("Reading dictionary file…");
    let file = brotli::Decompressor::new(
        std::fs::File::open("./polish.out.br").unwrap(),
        4096, /* buffer size */
    );
    let dict = std::io::BufReader::new(file).lines().par_bridge();

    println!("Building dictionary HashMap…");
    let result = dict
        .fold(
            || HashMap::new(),
            |mut acc: HashMap<String, String>, line| {
                let y = line.expect("Something went wrong");
                let x = y.split(';').take(2).collect::<Vec<&str>>();
                acc.insert(x[1].to_string(), x[0].to_string());
                acc
            },
        )
        .reduce_with(|mut left, right| {
            right.into_iter().for_each(|(k, v)| {
                left.insert(k, v);
            });
            left
        })
        .unwrap();

    result
}
