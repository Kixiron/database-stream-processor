use dbsp::{
    circuit::Root,
    operator::{DelayedFeedback, Generator},
    time::NestedTimestamp32,
    trace::{ord::OrdZSet, Batch, BatchReader, Cursor},
};
use std::{cell::RefCell, mem::take, rc::Rc};

type Weight = isize;

fn main() {
    // These are the numbers we want to calculate fibonacci for
    let queries = &[0, 1, 2, 3, 4, 5, 6, 7, 10, 20, 2000];

    // Calculate fibonacci for all of the given queries
    let (answers, too_high) = fibonacci(queries);

    dbg!(answers, too_high);
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
/// NeedsFib(n) :- Queries(n), n <= 93.
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
fn fibonacci(queries: &[u64]) -> (Vec<(u64, u64)>, Vec<u64>) {
    let answers = Rc::new(RefCell::new(Vec::new()));
    let too_high = Rc::new(RefCell::new(Vec::new()));

    // Make the queries we're given into a set
    let queries: OrdZSet<_, Weight> = OrdZSet::from_tuples(
        (),
        queries
            .iter()
            .copied()
            .map(|query| ((query, ()), 1))
            .collect(),
    );

    let (answers_copy, too_high_copy) = (answers.clone(), too_high.clone());
    let root = Root::build(move |circuit| {
        // Create a source that yields the queried values
        let queries = circuit.add_source(Generator::new(move || queries.clone()));

        // Get all the queries that are too high to fit in a u64
        queries
            .filter_keys::<OrdZSet<_, _>, _>(|&n| n > 93)
            .inspect(move |rejected| {
                let mut too_high = too_high_copy.borrow_mut();

                let mut cursor = rejected.cursor();
                while cursor.key_valid() {
                    too_high.push(*cursor.key());
                    cursor.step_key();
                }
            });

        // Create the `NeedsFib` relation
        let needs_fib = circuit
            .fixedpoint(|child| {
                let needs_fib = DelayedFeedback::<_, OrdZSet<u64, Weight>>::new(child);

                let n_minus_one = needs_fib.stream().filter_map_keys(|&n| n.checked_sub(1));
                let n_minus_two = needs_fib.stream().filter_map_keys(|&n| n.checked_sub(2));

                // Combine `Queries`, `NeedsFib(n - 1)` and `NeedsFib(n - 2)` into one stream
                let n = queries.delta0(child).filter_keys(|&n| n <= 93);
                let sum = n.sum([&n_minus_one, &n_minus_two]);
                // Connect the sum of the streams to the feedback variable
                needs_fib.connect(&sum);

                // Export the final stream from the sub-scope
                Ok(sum.integrate_trace().export())
            })
            .expect("failed to construct NeedsFib")
            .consolidate::<OrdZSet<_, _>>()
            .distinct();

        // Fib(0, 0).
        // Fib(1, 1).
        let base_cases = circuit.add_source(Generator::new(|| {
            OrdZSet::from_tuples((), vec![(((0, 0), ()), 1), (((1, 1), ()), 1)])
        }));

        // Fib(n, fib1 + fib2) :-
        //     NeedsFib(n),
        //     Fib(n - 1, fib1),
        //     Fib(n - 2, fib2).
        let fib = circuit
            .fixedpoint(|child| {
                let fib = DelayedFeedback::<_, OrdZSet<(u64, u64), Weight>>::new(child);
                let fib_trace = fib
                    .stream()
                    .plus(&base_cases.delta0(child))
                    .index::<u64, u64>();

                // fib.stream().inspect(|trace| {
                //     let mut cursor = trace.cursor();
                //     while cursor.key_valid() {
                //         let key = *cursor.key();
                //         while cursor.val_valid() {
                //             println!("fib_var: ({key}, {})", cursor.val());
                //             cursor.step_val();
                //         }
                //         cursor.step_key();
                //     }
                // });

                // Import `NeedsFib` into the sub-scope
                let needs_fib = needs_fib.delta0(child);

                // Create `NeedsFib(n - 1)` and `NeedsFib(n - 2)`
                let needs_fib_sub_one = needs_fib.index_with(|&n| {
                    // println!(
                    //     "needs_fib_sub_one: ({n} - 1 = {}, {n})",
                    //     n.saturating_sub(1),
                    // );
                    (n.saturating_sub(1), n)
                });
                let needs_fib_sub_two = needs_fib.index_with(|&n| {
                    // println!(
                    //     "needs_fib_sub_two: ({n} - 2 = {}, {n})",
                    //     n.saturating_sub(2),
                    // );
                    (n.saturating_sub(2), n)
                });

                let fib_sub_one = needs_fib_sub_one
                    .join_trace::<NestedTimestamp32, _, _, OrdZSet<_, _>>(
                        &fib_trace,
                        |_, &n, &fib| {
                            // println!("fib_sub_one: ({n}, fib({fib_sub_one}) = {fib})");
                            (n, fib)
                        },
                    )
                    .index();
                let fib_sub_two = needs_fib_sub_two
                    .join_trace::<NestedTimestamp32, _, _, OrdZSet<_, _>>(
                        &fib_trace,
                        |_, &n, &fib| {
                            // println!("fib_sub_two: ({n}, fib({n_sub_two}) = {fib})");
                            (n, fib)
                        },
                    )
                    .index();

                let fib_for_n = fib_sub_one.join(&fib_sub_two, |&n, &fib1, &fib2| {
                    println!("fib_for_n: ({n}, {fib1} + {fib2} = {})", fib1 + fib2);
                    (n, fib1 + fib2)
                });

                let sum = fib.stream().plus(&fib_for_n).distinct();
                fib.connect(&sum);

                Ok(sum.integrate_trace().export())
            })
            .expect("failed to construct Fib circuit")
            .consolidate::<OrdZSet<_, _>>()
            .index();

        fib.inspect(|trace| {
            let mut cursor = trace.cursor();
            while cursor.key_valid() {
                let key = *cursor.key();
                while cursor.val_valid() {
                    println!("fib: ({key}, {})", cursor.val());
                    cursor.step_val();
                }
                cursor.step_key();
            }
        });

        let query_trace = queries.index_with(|&n| (n, ()));
        fib.join_trace::<(), _, _, OrdZSet<_, _>>(&query_trace, |&n, &fib, &()| (n, fib))
            .inspect(move |values| {
                let mut answers = answers_copy.borrow_mut();

                let mut cursor = values.cursor();
                while cursor.key_valid() {
                    answers.push(*cursor.key());
                    cursor.step_key();
                }
            });
    })
    .expect("failed to build circuit");

    root.step().expect("failed to step root circuit");

    let mut answers = answers.borrow_mut();
    let mut too_high = too_high.borrow_mut();
    (take(&mut *answers), take(&mut *too_high))
}
