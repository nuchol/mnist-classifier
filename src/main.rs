use std::fs::File;
use std::error::Error;

use csv::ReaderBuilder;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn read_csv(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;

    for result in rdr.records().take(1) {
        let record = result?;
        cols = cols.max(record.len());

        for item in record.iter() {
            data.push(item.parse::<f32>()?);
        }

        rows += 1;
    }

    Ok(Array2::from_shape_vec((rows, cols), data)?.reversed_axes())
}

fn main() {
    const L1_SIZE: usize = 10;
    let data = read_csv("res/mnist_train.csv").unwrap();
    let (img_size, num_imgs) = data.dim();

    let w_1 = Array::random((L1_SIZE, img_size), Uniform::new(0.,1.).unwrap());
    let b_1 = Array::random((L1_SIZE, num_imgs), Uniform::new(0.,1.).unwrap());
    let z_1 = w_1.dot(&data) + &b_1;

    println!("======= Data =======\n{:?}\n", data);
    println!("======= W_1  =======\n{:?}\n", w_1);
    println!("======= b_1  =======\n{:?}\n", b_1);
    println!("======= Z_1  =======\n{:?}\n", z_1);
}
