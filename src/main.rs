use std::fs::File;
use std::error::Error;

use csv::ReaderBuilder;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn read_csv(path: &str) -> Result<Array2<u8>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;

    for result in rdr.records().take(1000) {
        let record = result?;
        cols = cols.max(record.len());

        for item in record.iter() {
            data.push(item.parse::<u8>()?);
        }

        rows += 1;
    }

    Ok(Array2::from_shape_vec((rows, cols), data)?.reversed_axes())
}

fn main() {
    const L1_SIZE: usize = 10;
    let data = read_csv("res/mnist_train.csv").unwrap();
    let img_size = data.dim().0;

    let w_1 = Array::random((L1_SIZE, img_size), Uniform::new(0.,1.).unwrap());
    println!("{:?}", w_1);

    println!("{:?}", data.dim());
    println!("{:?}", data.row(0));
}
