use blake3::Hasher as Blake3Hasher;
use clap::{PossibleValue, ValueEnum};
use core::slice;
use dbsp::{
    algebra::{HasOne, Present, F64},
    hash::{default_hash, default_hasher},
    trace::{layers::column_layer::ColumnLayer, Batch, Batcher, Builder},
    Circuit, OrdIndexedZSet, OrdZSet, Stream,
};
use indicatif::{HumanBytes, ProgressBar, ProgressState, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use reqwest::header::CONTENT_LENGTH;
use std::{
    cmp::Reverse,
    collections::HashMap,
    ffi::OsStr,
    fmt::{self, Debug},
    fs::{self, File, OpenOptions},
    hash::Hasher,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    mem::size_of,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};
use tar::Archive;
use zstd::Decoder;

pub type Node = u64;
/// Pagerank must use 64bit float values
pub type Rank = F64;
pub type Vertex = u64;
pub type Weight = isize;
pub type Distance = u64;

pub type VertexSet<D = Present> = OrdZSet<Node, D>;
pub type RankMap = OrdIndexedZSet<Node, Rank, Weight>;
pub type EdgeMap<D = Present> = OrdIndexedZSet<Node, Node, D>;
pub type DistanceSet<D = Present> = OrdZSet<(Node, Distance), D>;
pub type DistanceMap<D = Present> = OrdIndexedZSet<Node, Distance, D>;

pub type Streamed<P, T> = Stream<Circuit<P>, T>;

pub type Ranks<P> = Streamed<P, RankMap>;
pub type Edges<P, D = Present> = Streamed<P, EdgeMap<D>>;
pub type Vertices<P, D = Present> = Streamed<P, VertexSet<D>>;

type LoadedDataset<R> = (
    Properties,
    Vec<EdgeMap>,
    Vec<VertexSet>,
    <R as ResultParser>::Parsed,
);

const DATA_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/benches/ldbc-graphalytics-data",
);
const CHECKSUM_CONTEXT: &str = "DBSP LDBC Graphalyics Benchmark 2022-11-20 12:10:51 \
    producing checksums for optimized datasets";

pub(crate) fn optimize_dataset(dataset: DataSet) {
    let path = dataset.path();
    let properties = dataset.properties();

    print!("loading vertices for {}... ", dataset.name);
    io::stdout().flush().unwrap();
    let start = Instant::now();

    let vertex_data = VertexParser::new(File::open(path.join(&properties.vertex_file)).unwrap())
        .load_deduped(properties.vertices as usize);

    let elapsed = start.elapsed();
    println!("finished in {elapsed:#?}");

    let vertex_path = path.join(format!("{}.bin", properties.vertex_file));
    let vertex_checksum_path = path.join(format!("{}.bin.checksum", properties.vertex_file));
    write_optimized_dataset(&vertex_path, &vertex_checksum_path, |writer| {
        writer
            .write_all(&(vertex_data.len() as u64).to_le_bytes())
            .unwrap();

        if cfg!(target_endian = "little") {
            let vertex_bytes: &[u8] = unsafe {
                slice::from_raw_parts(
                    vertex_data.as_ptr().cast(),
                    vertex_data.len() * size_of::<u64>(),
                )
            };

            writer.write_all(vertex_bytes).unwrap();
        } else {
            for vertex in &vertex_data {
                writer.write_all(&vertex.to_le_bytes()).unwrap();
            }
        }
    });
    drop(vertex_data);

    print!("loading edges for {}... ", dataset.name);
    io::stdout().flush().unwrap();
    let start = Instant::now();

    let edge_data = EdgeParser::new(
        File::open(path.join(&properties.edge_file)).unwrap(),
        properties.directed,
    )
    .load_deduped(properties.edges as usize);

    let elapsed = start.elapsed();
    println!("finished in {elapsed:#?}");

    let edge_path = path.join(format!("{}.bin", properties.edge_file));
    let edge_checksum_path = path.join(format!("{}.bin.checksum", properties.edge_file));
    write_optimized_dataset(&edge_path, &edge_checksum_path, |writer| {
        writer
            .write_all(&(edge_data.len() as u64).to_le_bytes())
            .unwrap();

        if cfg!(target_endian = "little") {
            let edge_bytes: &[u8] = unsafe {
                slice::from_raw_parts(
                    edge_data.as_ptr().cast(),
                    edge_data.len() * size_of::<[u64; 2]>(),
                )
            };

            writer.write_all(edge_bytes).unwrap();
        } else {
            for [src, dest] in &edge_data {
                writer.write_all(&src.to_le_bytes()).unwrap();
                writer.write_all(&dest.to_le_bytes()).unwrap();
            }
        }
    });
    drop(edge_data);
}

fn write_optimized_dataset<F>(data_path: &Path, checksum_path: &Path, write_data: F)
where
    F: FnOnce(&mut dyn Write),
{
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(data_path)
        .unwrap();

    print!(
        "started writing optimized version of {}... ",
        data_path.display(),
    );
    io::stdout().flush().unwrap();
    let start = Instant::now();

    // Write the optimized data to the file
    let mut writer = BufWriter::new(&mut file);
    write_data(&mut writer);
    writer.flush().unwrap();
    drop(writer);

    let elapsed = start.elapsed();
    println!("finished in {elapsed:#?}");

    print!(
        "started computing checksum of of {}... ",
        data_path.display(),
    );
    io::stdout().flush().unwrap();
    let start = Instant::now();

    // Checksum the file
    // TODO: Advisory file locking for mmap-ing
    // TODO: madvise() could be potentially useful here
    // https://docs.rs/memmap2/latest/memmap2/struct.Mmap.html#method.advise
    let contents = unsafe { Mmap::map(&file).unwrap() };
    let checksum = hash_contents(&contents);

    let mut checksum_file = File::create(checksum_path).unwrap();
    checksum_file.write_all(&checksum).unwrap();
    checksum_file.flush().unwrap();

    let elapsed = start.elapsed();
    println!("finished in {elapsed:#?}");
}

fn hash_contents(contents: &[u8]) -> [u8; 256] {
    let mut hasher = Blake3Hasher::new_derive_key(CHECKSUM_CONTEXT);

    // `Hasher::update_rayon()` only yields performance benefits for buffers longer
    // than 128KiB
    if contents.len() > 1 << 17 {
        hasher.update_rayon(contents);
    } else {
        hasher.update(contents);
    }

    let mut checksum = [0; 256];
    hasher.finalize_xof().fill(&mut checksum);
    checksum
}

pub(crate) fn list_downloaded_benchmarks() {
    let data_path = Path::new(DATA_PATH);

    let mut datasets = Vec::new();
    for dir in fs::read_dir(data_path)
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_type().map_or(false, |ty| ty.is_dir()))
    {
        let path = dir.path();

        if let Ok(dir) = fs::read_dir(path) {
            for entry in dir.flatten() {
                let path = entry.path();
                if path.extension() == Some(OsStr::new("properties")) {
                    let properties_file = File::open(&path).unwrap_or_else(|error| {
                        panic!("failed to open {}: {error}", path.display())
                    });

                    let name = path
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .split_once('.')
                        .unwrap()
                        .0
                        .to_owned();

                    let properties = Properties::from_file(&name, properties_file);
                    let vertex_bytes = properties.vertices * size_of::<Node>() as u64;
                    let edge_bytes = properties.edges * size_of::<Node>() as u64 * 2;

                    datasets.push((name, properties.scale(), vertex_bytes, edge_bytes));
                    break;
                }
            }
        }
    }
    datasets.sort_by_key(|(.., vertex_bytes, edge_bytes)| Reverse(vertex_bytes + edge_bytes));

    if datasets.is_empty() {
        println!("No datasets are currently downloaded");
    }

    let longest_name = datasets.iter().map(|(name, ..)| name.len()).max().unwrap() + 1;

    let mut stdout = io::stdout().lock();
    for (name, scale, vertex_bytes, edge_bytes) in datasets {
        writeln!(
            stdout,
            "{name:<longest_name$} scale: {:.01}, total size: {}, vertices: {}, edges: {}",
            scale,
            HumanBytes(vertex_bytes + edge_bytes),
            HumanBytes(vertex_bytes),
            HumanBytes(edge_bytes),
        )
        .unwrap();
    }

    stdout.flush().unwrap();
}

pub(crate) fn list_datasets() {
    let cache_file = Path::new(DATA_PATH).join("dataset_cache.json");
    let dataset_sizes = if cache_file.exists() {
        serde_json::from_reader(File::open(&cache_file).unwrap()).unwrap_or_default()
    } else {
        let mut sizes = HashMap::with_capacity(DataSet::DATASETS.len());

        // TODO: Realistically we should be doing all of these requests in parallel but
        // I don't feel like adding tokio as a direct dependency at the moment (it's
        // already a transitive dependency so it doesn't *really* matter, I'm just lazy)
        let client = reqwest::blocking::Client::new();
        for dataset in DataSet::DATASETS {
            if let Ok(response) = client.head(dataset.url).send() {
                if let Some(length) = response.headers()[CONTENT_LENGTH]
                    .to_str()
                    .ok()
                    .and_then(|len| len.parse::<u64>().ok())
                {
                    sizes.insert(dataset.name.to_owned(), length);
                }
            }
        }

        fs::create_dir_all(DATA_PATH).unwrap();
        let cache_file = BufWriter::new(File::create(&cache_file).unwrap());
        serde_json::to_writer_pretty(cache_file, &sizes).unwrap();

        sizes
    };

    let mut datasets = DataSet::DATASETS.to_vec();
    datasets.sort_by_key(|dataset| (dataset.scale, dataset_sizes.get(dataset.name).copied()));

    let longest_name = datasets
        .iter()
        .map(|dataset| dataset.name.len())
        .max()
        .unwrap();

    let mut stdout = io::stdout().lock();
    for dataset in datasets {
        write!(
            stdout,
            "{:<longest_name$} scale: {:?} archive size: ",
            dataset.name, dataset.scale,
        )
        .unwrap();

        if let Some(&length) = dataset_sizes.get(dataset.name) {
            writeln!(stdout, "{}", HumanBytes(length)).unwrap();
        } else {
            writeln!(stdout, "???").unwrap();
        }
    }

    stdout.flush().unwrap();
}

#[derive(Debug, Clone, Copy)]
pub struct DataSet {
    pub name: &'static str,
    pub url: &'static str,
    pub scale: Scale,
}

impl DataSet {
    pub const fn new(name: &'static str, url: &'static str, scale: Scale) -> Self {
        Self { name, url, scale }
    }

    pub fn path(&self) -> PathBuf {
        Path::new(DATA_PATH).join(self.name)
    }

    pub fn properties(&self) -> Properties {
        let dataset_dir = self.dataset_dir().unwrap();

        // Open & parse the properties file
        let properties_path = dataset_dir.join(format!("{}.properties", self.name));
        let properties_file = File::open(&properties_path).unwrap_or_else(|error| {
            panic!("failed to open {}: {error}", properties_path.display())
        });

        Properties::from_file(self.name, properties_file)
    }

    pub fn load<R: ResultParser>(&self, workers: usize) -> io::Result<LoadedDataset<R>> {
        let dataset_dir = self.dataset_dir()?;

        // Open & parse the properties file
        let properties = self.properties();

        // Load the edges and vertices in parallel
        let (data_dir, props) = (dataset_dir.clone(), properties.clone());
        let edges_handle = thread::spawn(move || {
            let edges_path = data_dir.join(&props.edge_file);
            let optimized_edges_path = data_dir.join(format!("{}.bin", props.edge_file));
            let optimized_edges_checksum_path =
                data_dir.join(format!("{}.bin.checksum", props.edge_file));

            if optimized_edges_path.exists() && optimized_edges_checksum_path.exists() {
                let edges = File::open(&optimized_edges_path).unwrap_or_else(|error| {
                    panic!("failed to open {}: {error}", optimized_edges_path.display())
                });
                let contents = unsafe { Mmap::map(&edges).unwrap() };
                let expected_checksum = hash_contents(&contents);

                let mut checksum_file = File::open(&optimized_edges_checksum_path).unwrap();
                let mut current_checksum = [0; 256];
                checksum_file.read_exact(&mut current_checksum).unwrap();

                if current_checksum == expected_checksum {
                    return Self::load_optimized_edges(&contents, workers, props.directed);
                } else {
                    eprintln!(
                        "checksum for {} doesn't match the file's contents",
                        optimized_edges_path.display(),
                    );
                }
            }

            let edges = File::open(&edges_path)
                .unwrap_or_else(|error| panic!("failed to open {}: {error}", edges_path.display()));
            EdgeParser::new(edges, props.directed).load(props.edges as usize, workers)
        });

        // Open the vertices file
        let vertices_path = dataset_dir.join(&properties.vertex_file);
        let optimized_vertices_path = dataset_dir.join(format!("{}.bin", properties.vertex_file));
        let optimized_vertices_checksum_path =
            dataset_dir.join(format!("{}.bin.checksum", properties.vertex_file));

        let (vertices, vertices_optimized) =
            if optimized_vertices_path.exists() && optimized_vertices_checksum_path.exists() {
                let file = File::open(&optimized_vertices_path).unwrap_or_else(|error| {
                    panic!("failed to open {}: {error}", vertices_path.display())
                });
                let contents = unsafe { Mmap::map(&file).unwrap() };
                let expected_checksum = hash_contents(&contents);

                let mut checksum_file = File::open(&optimized_vertices_checksum_path).unwrap();
                let mut current_checksum = [0; 256];
                checksum_file.read_exact(&mut current_checksum).unwrap();

                if current_checksum == expected_checksum {
                    (file, true)
                } else {
                    eprintln!(
                        "checksum for {} doesn't match the file's contents",
                        optimized_vertices_path.display(),
                    );
                    let file = File::open(&vertices_path).unwrap_or_else(|error| {
                        panic!("failed to open {}: {error}", vertices_path.display())
                    });
                    (file, false)
                }
            } else {
                let file = File::open(&vertices_path).unwrap_or_else(|error| {
                    panic!("failed to open {}: {error}", vertices_path.display())
                });
                (file, false)
            };

        let (vertices, results) = if let Some(suffix) = R::file_suffix() {
            let result_path = dataset_dir.join(format!("{}{suffix}", self.name));
            let props = properties.clone();
            let results_handle = thread::spawn(move || {
                // Open the results file
                let result_file = File::open(&result_path).unwrap_or_else(|error| {
                    panic!("failed to open {}: {error}", result_path.display())
                });

                // Parse the results file in parallel to the vertices file
                R::load(&props, result_file)
            });

            let vertices = if vertices_optimized {
                Self::load_optimized_vertices(vertices, workers)
            } else {
                VertexParser::new(vertices).load(properties.vertices as usize, workers)
            };
            let results = results_handle.join().unwrap();

            (vertices, results)

        // Otherwise parse the vertices file on this thread
        } else {
            let vertices = if vertices_optimized {
                Self::load_optimized_vertices(vertices, workers)
            } else {
                VertexParser::new(vertices).load(properties.vertices as usize, workers)
            };

            (vertices, R::Parsed::default())
        };

        // Wait for the vertex and edge threads to finish parsing
        let edges = edges_handle.join().unwrap();

        Ok((properties, edges, vertices, results))
    }

    fn load_optimized_vertices(file: File, workers: usize) -> Vec<OrdZSet<u64, Present>> {
        let contents = unsafe { Mmap::map(&file).unwrap() };
        let length = u64::from_le_bytes(contents[..size_of::<u64>()].try_into().unwrap()) as usize;
        let vertices = unsafe {
            slice::from_raw_parts(
                contents.as_ptr().add(size_of::<u64>()).cast::<u64>(),
                length,
            )
        };

        if workers == 1 {
            let mut keys = vertices.to_vec();

            // TODO: Vectorized bswap
            if cfg!(target_endian = "big") {
                for key in &mut keys {
                    *key = key.swap_bytes();
                }
            }

            let diffs = vec![Present; keys.len()];
            vec![OrdZSet::from_layer(unsafe {
                ColumnLayer::from_parts(keys, diffs)
            })]
        } else {
            let mut keys: Vec<_> = (0..workers)
                .map(|_| Vec::with_capacity(length / workers))
                .collect();

            let mut hasher = default_hasher();
            for &vertex in vertices {
                let vertex = if cfg!(target_endian = "little") {
                    vertex
                } else {
                    vertex.swap_bytes()
                };

                hasher.write_u64(vertex);
                let shard = hasher.finish() as usize % workers;
                keys[shard].push(vertex);
                hasher.reset();
            }

            keys.into_iter()
                .map(|mut keys| {
                    keys.shrink_to_fit();

                    let diffs = vec![Present; keys.len()];
                    OrdZSet::from_layer(unsafe { ColumnLayer::from_parts(keys, diffs) })
                })
                .collect()
        }
    }

    fn load_optimized_edges(contents: &[u8], workers: usize, directed: bool) -> Vec<EdgeMap> {
        let length = u64::from_le_bytes(contents[..size_of::<u64>()].try_into().unwrap()) as usize;
        let vertices = unsafe {
            slice::from_raw_parts(
                contents.as_ptr().add(size_of::<u64>()).cast::<[u64; 2]>(),
                length,
            )
        };

        if directed {
            let mut edges: Vec<_> = (0..workers)
                .map(|_| <EdgeMap as Batch>::Builder::with_capacity((), length / workers))
                .collect();

            let mut hasher = default_hasher();
            for &vertex in vertices {
                let [src, dest] = if cfg!(target_endian = "little") {
                    vertex
                } else {
                    [vertex[0].swap_bytes(), vertex[1].swap_bytes()]
                };

                hasher.write_u64(src);
                let shard = hasher.finish() as usize % workers;
                edges[shard].push(((src, dest), Present));
                hasher.reset();
            }

            edges.into_iter().map(Builder::done).collect()
        } else {
            let mut batches: Vec<_> = (0..workers)
                .map(|_| {
                    (
                        Vec::with_capacity(length / workers / 2),
                        Vec::with_capacity(length / workers / 2),
                    )
                })
                .collect();

            let mut hasher = default_hasher();
            for &vertex in vertices {
                let [src, dest] = if cfg!(target_endian = "little") {
                    vertex
                } else {
                    [vertex[0].swap_bytes(), vertex[1].swap_bytes()]
                };

                hasher.write_u64(src);
                let shard = hasher.finish() as usize % workers;
                batches[shard].0.push(((src, dest), Present));
                hasher.reset();

                hasher.write_u64(dest);
                let shard = hasher.finish() as usize % workers;
                batches[shard].1.push(((dest, src), Present));
                hasher.reset();
            }

            let mut edges = Vec::with_capacity(batches.len());
            batches
                .into_par_iter()
                .map(|(mut forward, mut reverse)| {
                    let mut edges = <EdgeMap as Batch>::Batcher::new_batcher(());
                    edges.push_consolidated_batch(&mut forward);
                    reverse.sort_unstable_by_key(|&((dest, _), _)| dest);
                    edges.push_consolidated_batch(&mut reverse);
                    edges.seal()
                })
                .collect_into_vec(&mut edges);

            edges
        }
    }

    pub fn load_results<R: ResultParser>(&self, props: &Properties) -> io::Result<R::Parsed> {
        if let Some(suffix) = R::file_suffix() {
            let dataset_dir = self.dataset_dir()?;

            let result_path = dataset_dir.join(format!("{}{suffix}", self.name));
            let result_file = File::open(&result_path).unwrap_or_else(|error| {
                panic!("failed to open {}: {error}", result_path.display())
            });

            Ok(R::load(props, result_file))
        } else {
            Ok(R::Parsed::default())
        }
    }

    /// Gets the dataset's directory if it exists or downloads and extracts it
    ///
    /// The full data repository is stored [here], the downloads can be *very*
    /// slow
    ///
    /// [here]: https://repository.surfsara.nl/datasets/cwi/graphalytics
    fn dataset_dir(&self) -> io::Result<PathBuf> {
        let data_path = self.path();
        let archive_path = Path::new(DATA_PATH).join(format!("{}.tar.zst", self.name));
        let tarball_path = Path::new(DATA_PATH).join(format!("{}.tar", self.name));

        fs::create_dir_all(&data_path)?;

        // If it doesn't exist, download the dataset
        // TODO: Check if dir is empty
        if !(archive_path.exists() || tarball_path.exists()) {
            let mut archive_file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&archive_path)
                .unwrap_or_else(|error| {
                    panic!("failed to create {}: {error}", archive_path.display())
                });
            let mut writer = BufWriter::new(&mut archive_file);

            // Download and write the archive to disk
            println!(
                "\ndownloading {} from {}, this may take a while",
                self.name, self.url
            );
            let response = reqwest::blocking::get(self.url)
                .unwrap_or_else(|error| panic!("failed to download {}: {error}", self.url));

            let progress = if let Some(content_length) = response.content_length() {
                let progress = ProgressBar::new(content_length);
                progress.enable_steady_tick(Duration::from_millis(300));
                progress.set_style(
                    ProgressStyle::with_template(
                        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                    )
                    .unwrap()
                    .with_key("eta", |state: &ProgressState, write: &mut dyn fmt::Write| write!(write, "{:.1}s", state.eta().as_secs_f64()).unwrap())
                    .progress_chars("#>-"),
                );

                progress
            } else {
                todo!()
            };

            let mut response = BufReader::new(response);
            loop {
                let chunk = response.fill_buf()?;
                // `.fill_buf()` returns an empty slice when the underlying reader is done
                if chunk.is_empty() {
                    break;
                }

                writer.write_all(chunk)?;

                // Consume the chunk's bytes
                let chunk_len = chunk.len();
                progress.inc(chunk_len as u64);
                response.consume(chunk_len);
            }

            // Flush the writer
            writer
                .flush()
                .unwrap_or_else(|error| panic!("failed to flush {} to disk: {error}", self.url));
            progress.finish_with_message("done");
        }

        if !tarball_path.exists() && fs::read_dir(&data_path).unwrap().count() == 0 {
            // Note that we're *opening* the file and not *creating* it
            let archive_file = BufReader::new(File::open(&archive_path).unwrap_or_else(|error| {
                panic!("failed to create {}: {error}", archive_path.display())
            }));

            // Decompress the zstd-compressed tarball
            let mut decoder = Decoder::new(archive_file)?;
            let mut tarball = BufWriter::new(File::create(&tarball_path)?);
            io::copy(&mut decoder, &mut tarball)?;

            // TODO: Maybe want to delete the original zsd file?
        }

        // TODO: Finer-grained check for the files we care about
        if fs::read_dir(&data_path).unwrap().count() == 0 {
            // Note that we're *opening* the file and not *creating* it
            let archive_file = BufReader::new(File::open(&tarball_path).unwrap_or_else(|error| {
                panic!("failed to create {}: {error}", archive_path.display())
            }));

            // Open the archive
            let mut tar_archive = Archive::new(archive_file);

            // Extract the archive
            println!(
                "extracting {} to {}",
                archive_path.display(),
                data_path.display(),
            );
            tar_archive.unpack(&data_path).unwrap_or_else(|error| {
                panic!(
                    "failed to extract '{}' to '{}': {error}",
                    archive_path.display(),
                    data_path.display(),
                )
            });

            // TODO: Maybe want to delete the original tarball?
        }

        Ok(data_path)
    }
}

macro_rules! datasets {
    ($($const:ident = $name:literal @ $scale:ident),* $(,)?) => {
        const DATASETS_LEN: usize = [$(DataSet::$const,)*].len();

        impl DataSet {
            pub const DATASETS: [Self; DATASETS_LEN] = [$(Self::$const,)*];

            $(
                pub const $const: Self = Self::new(
                    $name,
                    concat!("https://r2-public-worker.ldbc.workers.dev/graphalytics/", $name, ".tar.zst"),
                    Scale::$scale,
                );
            )*
        }
    }
}

datasets! {
    EXAMPLE_DIR = "example-directed" @ Example,
    EXAMPLE_UNDIR = "example-undirected" @ Example,

    DATAGEN_7_5 = "datagen-7_5-fb" @ S,
    DATAGEN_7_6 = "datagen-7_6-fb" @ S,
    DATAGEN_7_7 = "datagen-7_7-zf" @ S,
    DATAGEN_7_8 = "datagen-7_8-zf" @ S,
    DATAGEN_7_9 = "datagen-7_9-fb" @ S,

    DATAGEN_8_0 = "datagen-8_0-fb" @ M,
    DATAGEN_8_1 = "datagen-8_1-fb" @ M,
    DATAGEN_8_2 = "datagen-8_2-zf" @ M,
    DATAGEN_8_3 = "datagen-8_3-zf" @ M,
    DATAGEN_8_4 = "datagen-8_4-fb" @ M,
    DATAGEN_8_5 = "datagen-8_5-fb" @ L,
    DATAGEN_8_6 = "datagen-8_6-fb" @ L,
    DATAGEN_8_7 = "datagen-8_7-zf" @ L,
    DATAGEN_8_8 = "datagen-8_8-zf" @ L,
    DATAGEN_8_9 = "datagen-8_9-fb" @ L,

    DATAGEN_9_0 = "datagen-9_0-fb" @ XL,
    DATAGEN_9_1 = "datagen-9_1-fb" @ XL,
    DATAGEN_9_2 = "datagen-9_2-zf" @ XL,
    DATAGEN_9_3 = "datagen-9_3-zf" @ XL,
    DATAGEN_9_4 = "datagen-9_4-fb" @ XL,

    DATAGEN_SF3K = "datagen-sf3k-fb" @ XL,
    DATAGEN_SF10K = "datagen-sf10k-fb" @ XL,

    GRAPH_500_22 = "graph500-22" @ S,
    GRAPH_500_23 = "graph500-23" @ M,
    GRAPH_500_24 = "graph500-24" @ M,
    GRAPH_500_25 = "graph500-25" @ L,
    GRAPH_500_26 = "graph500-26" @ XL,
    GRAPH_500_27 = "graph500-27" @ XL,
    GRAPH_500_28 = "graph500-28" @ XXL,
    GRAPH_500_29 = "graph500-29" @ XXL,
    GRAPH_500_30 = "graph500-30" @ XXL,

    KGS = "kgs" @ XS,
    WIKI_TALK = "wiki-Talk" @ XXS,
    CIT_PATENTS = "cit-Patents" @ XS,
    DOTA_LEAGUE = "dota-league" @ S,
    TWITTER_MPI = "twitter_mpi" @ XL,
    COM_FRIENDSTER = "com-friendster" @ XL,
}

impl ValueEnum for DataSet {
    fn value_variants<'a>() -> &'a [Self] {
        &Self::DATASETS
    }

    fn to_possible_value<'a>(&self) -> Option<PossibleValue<'a>> {
        Some(PossibleValue::new(self.name))
    }
}

impl Default for DataSet {
    fn default() -> Self {
        Self::EXAMPLE_DIR
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(clippy::upper_case_acronyms)]
pub enum Scale {
    /// Really tiny datasets for testing
    Example,
    /// Datasets with a scale factor of 6.5 to 6.9
    XXS,
    /// Datasets with a scale factor of 7.0 to 7.4
    XS,
    /// Datasets with a scale factor of 7.5 to 7.9
    S,
    /// Datasets with a scale factor of 8.0 to 8.4
    M,
    /// Datasets with a scale factor of 8.5 to 8.9
    L,
    /// Datasets with a scale factor of 9.0 to 9.4
    XL,
    /// Datasets with a scale factor of 9.5 to 9.9
    XXL,
    // /// Datasets with a scale factor of 10.0 to 10.4
    // XXXL,
}

#[derive(Debug, Clone, Default)]
pub struct Properties {
    pub vertex_file: String,
    pub edge_file: String,
    pub vertices: u64,
    pub edges: u64,
    pub directed: bool,
    pub source_vertex: Vertex,
    pub algorithms: Vec<Algorithm>,
    pub pagerank_damping_factor: Option<f64>,
    pub pagerank_iters: Option<usize>,
}

impl Properties {
    pub fn from_file(dataset: &str, file: File) -> Self {
        let mut vertex_file = None;
        let mut edge_file = None;
        let mut vertices = None;
        let mut edges = None;
        let mut directed = None;
        let mut source_vertex = None;
        let mut algorithms = Vec::new();
        let mut pagerank_iters = None;
        let mut pagerank_damping_factor = None;

        let mut file = BufReader::new(file);
        let mut buffer = String::with_capacity(256);

        while let Ok(n) = file.read_line(&mut buffer) {
            if n == 0 {
                break;
            }
            let line = buffer.trim();

            if !(line.starts_with('#') || line.is_empty()) {
                // Remove `graph.{dataset}.` from every property
                let line = line
                    .trim_start_matches("graph.")
                    .trim_start_matches(dataset)
                    .trim_start_matches('.');

                let (_, value) = line.split_once('=').unwrap();
                let value = value.trim();

                if line.starts_with("bfs.source-vertex") {
                    source_vertex = Some(value.parse().unwrap());
                } else if line.starts_with("directed") {
                    directed = Some(value.parse().unwrap());
                } else if line.starts_with("vertex-file") {
                    vertex_file = Some(value.to_owned());
                } else if line.starts_with("edge-file") {
                    edge_file = Some(value.to_owned());
                } else if line.starts_with("meta.vertices") {
                    vertices = Some(value.parse().unwrap());
                } else if line.starts_with("meta.edges") {
                    edges = Some(value.parse().unwrap());
                } else if line.starts_with("algorithms") {
                    algorithms.extend(
                        value
                            .split(',')
                            .map(|algo| Algorithm::try_from(algo.trim()).unwrap()),
                    );
                } else if line.starts_with("pr.damping-factor") {
                    pagerank_damping_factor = Some(value.parse().unwrap());
                } else if line.starts_with("pr.num-iterations") {
                    pagerank_iters = Some(value.parse().unwrap());
                }
            }

            buffer.clear();
        }

        Self {
            vertex_file: vertex_file.unwrap(),
            edge_file: edge_file.unwrap(),
            vertices: vertices.unwrap(),
            edges: edges.unwrap(),
            directed: directed.unwrap(),
            source_vertex: source_vertex.unwrap(),
            algorithms,
            pagerank_damping_factor,
            pagerank_iters,
        }
    }

    /// Gets the scale of the current benchmark as defined
    /// [here](https://arxiv.org/pdf/2011.15028v4.pdf#subsection.2.2.3)
    pub fn scale(&self) -> f64 {
        (self.edges as f64 + self.vertices as f64).log10()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    Pr,
    Bfs,
    Lcc,
    Wcc,
    Cdlp,
    Sssp,
}

impl TryFrom<&str> for Algorithm {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match &*value.to_ascii_lowercase() {
            "pr" => Self::Pr,
            "bfs" => Self::Bfs,
            "lcc" => Self::Lcc,
            "wcc" => Self::Wcc,
            "cdlp" => Self::Cdlp,
            "sssp" => Self::Sssp,
            unknown => return Err(format!("unknown algorithm: {unknown:?}")),
        })
    }
}

struct EdgeParser {
    file: BufReader<File>,
    directed: bool,
}

impl EdgeParser {
    pub fn new(file: File, directed: bool) -> Self {
        Self {
            file: BufReader::new(file),
            directed,
        }
    }

    pub fn load_deduped(self, approx_edges: usize) -> Vec<[u64; 2]> {
        let mut edges = Vec::with_capacity(approx_edges);
        self.parse(|src, dest| edges.push([src, dest]));
        edges
    }

    pub fn load(self, approx_edges: usize, workers: usize) -> Vec<EdgeMap> {
        // Directed graphs can use an ordered builder
        if self.directed {
            let mut edges: Vec<_> = (0..workers)
                .map(|_| <EdgeMap as Batch>::Builder::with_capacity((), approx_edges / workers))
                .collect();

            let mut hasher = default_hasher();
            self.parse(|src, dest| {
                hasher.write_u64(src);
                let shard = hasher.finish() as usize % workers;
                edges[shard].push(((src, dest), Present));
                hasher.reset();
            });

            edges.into_iter().map(Builder::done).collect()

        // Undirected graphs must use an unordered builder
        } else {
            let mut batches: Vec<_> = (0..workers)
                .map(|_| {
                    (
                        Vec::with_capacity(approx_edges / workers / 2),
                        Vec::with_capacity(approx_edges / workers / 2),
                    )
                })
                .collect();

            let mut hasher = default_hasher();
            self.parse(|src, dest| {
                hasher.write_u64(src);
                let shard = hasher.finish() as usize % workers;
                batches[shard].0.push(((src, dest), Present));
                hasher.reset();

                hasher.write_u64(dest);
                let shard = hasher.finish() as usize % workers;
                batches[shard].1.push(((dest, src), Present));
                hasher.reset();
            });

            let mut edges = Vec::with_capacity(batches.len());
            batches
                .into_par_iter()
                .map(|(mut forward, mut reverse)| {
                    let mut edges = <EdgeMap as Batch>::Batcher::new_batcher(());
                    edges.push_consolidated_batch(&mut forward);
                    reverse.sort_unstable_by_key(|&((dest, _), _)| dest);
                    edges.push_consolidated_batch(&mut reverse);
                    edges.seal()
                })
                .collect_into_vec(&mut edges);

            edges
        }
    }

    fn parse<F>(mut self, mut append: F)
    where
        F: FnMut(Vertex, Vertex),
    {
        let mut buffer = String::with_capacity(256);
        while let Ok(n) = self.file.read_line(&mut buffer) {
            if n == 0 {
                break;
            }

            let line = buffer.trim_end();
            let mut line = line.splitn(3, ' ');

            let src = line.next().unwrap().parse().unwrap();
            let dest = line.next().unwrap().parse().unwrap();
            // let weight = line.next().and_then(|weight| weight.parse().ok());
            append(src, dest);

            buffer.clear();
        }
    }
}

struct VertexParser {
    file: BufReader<File>,
}

impl VertexParser {
    pub fn new(file: File) -> Self {
        Self {
            file: BufReader::new(file),
        }
    }

    pub fn load_deduped(self, approx_edges: usize) -> Vec<u64> {
        let mut vertices = Vec::with_capacity(approx_edges);
        self.parse(|vertex| vertices.push(vertex));
        vertices
    }

    pub fn load(self, approx_vertices: usize, workers: usize) -> Vec<VertexSet> {
        // The vertices file is ordered so we can use an ordered builder
        let mut vertices: Vec<_> = (0..workers)
            .map(|_| <VertexSet as Batch>::Builder::with_capacity((), approx_vertices / workers))
            .collect();

        self.parse(|vertex| {
            vertices[default_hash(&vertex) as usize % workers].push((vertex, Present))
        });

        vertices.into_iter().map(Builder::done).collect()
    }

    fn parse<F>(mut self, mut append: F)
    where
        F: FnMut(Vertex),
    {
        let mut buffer = String::with_capacity(256);
        while let Ok(n) = self.file.read_line(&mut buffer) {
            if n == 0 {
                break;
            }

            let line = buffer.trim_end();
            let vertex: Vertex = line.parse().unwrap();
            append(vertex);

            buffer.clear();
        }
    }
}

pub trait ResultParser {
    type Parsed: Default + Send + 'static;

    fn file_suffix() -> Option<&'static str>;

    fn load(props: &Properties, file: File) -> Self::Parsed;
}

pub struct NoopResults;

impl ResultParser for NoopResults {
    type Parsed = ();

    fn file_suffix() -> Option<&'static str> {
        None
    }

    fn load(_props: &Properties, _file: File) -> Self::Parsed {}
}

pub struct BfsResults;

impl ResultParser for BfsResults {
    type Parsed = DistanceSet<i8>;

    fn file_suffix() -> Option<&'static str> {
        Some("-BFS")
    }

    fn load(props: &Properties, file: File) -> Self::Parsed {
        let mut file = BufReader::new(file);

        // The bfs results file is ordered so we can use an ordered builder
        let mut results =
            <DistanceSet<i8> as Batch>::Builder::with_capacity((), props.vertices as usize);

        let mut buffer = String::with_capacity(256);
        while let Ok(n) = file.read_line(&mut buffer) {
            if n == 0 {
                break;
            }

            let line = buffer.trim_end();
            let (vertex, distance) = line.split_once(' ').unwrap();

            let vertex = vertex.parse().unwrap();
            let distance = distance.parse().unwrap();

            results.push(((vertex, distance), 1));
            buffer.clear();
        }

        results.done()
    }
}

pub struct PageRankResults;

impl ResultParser for PageRankResults {
    type Parsed = RankMap;

    fn file_suffix() -> Option<&'static str> {
        Some("-PR")
    }

    fn load(props: &Properties, file: File) -> Self::Parsed {
        let mut file = BufReader::new(file);

        // The pagerank results file is ordered so we can use an ordered builder
        let mut results = <RankMap as Batch>::Builder::with_capacity((), props.vertices as usize);

        let mut buffer = String::with_capacity(256);
        while let Ok(n) = file.read_line(&mut buffer) {
            if n == 0 {
                break;
            }

            let line = buffer.trim_end();
            let (vertex, distance) = line.split_once(' ').unwrap();

            let vertex = vertex.parse().unwrap();
            let rank = F64::new(distance.parse::<f64>().unwrap());

            results.push(((vertex, rank), Weight::one()));
            buffer.clear();
        }

        results.done()
    }
}
