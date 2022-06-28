use dbsp::circuit::{Root, Runtime};
use std::{env, thread};

fn main() {
    // Decide how many threads to run fibonacci on from the first argument given to us
    let threads = env::args()
        .nth(1)
        .and_then(|threads| threads.parse().ok())
        .or_else(|| thread::available_parallelism().ok())
        .map(|threads| threads.get())
        .unwrap_or(1);

    // These are the numbers we want to calculate fibonacci for
    let queries = &[0, 1, 2, 3, 4, 5, 6, 7, 10, 20];

    // Calculate fibonacci for all of the given queries
    let answers = fibonacci(threads, queries);
}

/// Calculates the fibonacci number for each number given in `queries`
/// by creating a dbsp program that uses `threads` threads
///
/// Creates the following program within dbsp:
/// ```text
/// #[input]
/// rel Queries(n: u64)
///
/// // Return all queries that are too high for us to
/// // calculate fibonacci numbers for (the highest fibonacci
/// // number that can fit in a u64 is `fib(93)`)
/// #[output]
/// rel TooHigh(n: u64)
/// TooHigh(n) :- Queries(n), n > 93.
///
/// // Figure out what numbers we need to calculate fibonacci for
/// rel NeedsFib(n: u64)
/// NeedsFib(n) :- Queries(n).
/// NeedsFib(n - 1) :- NeedsFib(n), n >= 1.
/// NeedsFib(n - 2) :- NeedsFib(n), n >= 2.
///
/// // The workhorse relation that actually calculates fibonacci numbers
/// rel Fib(n: u64, fib: u64)
///
/// // The base cases of zero and one
/// Fib(0, 1).
/// Fib(1, 1).
///
/// // Calculate fibonacci for everything in `NeedsFib`
/// Fib(n, fib1 + fib2) :-
///     NeedsFib(n),
///     Fib(n - 1, fib1),
///     Fib(n - 2, fib2).
///
/// #[output]
/// rel Answers(n: u64, fib: u64)
/// Answers(n, fib) :- Fib(n, fib), Queries(n).
/// ```
fn fibonacci(threads: usize, queries: &[u32]) {
    let runtime = Runtime::run(threads, |_runtime, _worker| {
        let root = Root::build(|circuit| {});
    });

    todo!()
}
