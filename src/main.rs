use rand::Rng;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::fs::File;
use rand_distr::Distribution;
const BINDINGCHANCE: f64 = 0.25; // Adjust this to control P_on
const RSQRD: f64 = 0.0025; 
const RUNS:usize = 100000;
const TARGET_RADIUS_SQRD: f64 = 0.0025;
const UNBINDCHANCE: f64 = 0.01; // Adjust this to control P_off

fn main() -> Result<(), Box<dyn std::error::Error>>{
    //Ready the writing part of the code.
    let file = OpenOptions::new()
        .append(true)  // Enable appending
        .create(true)  // Create the file if it doesn't exist
        .open("Data.txt")?;

    let mut PosWriter = BufWriter::new(file);  

    let TTDfile = OpenOptions::new()
    .append(true)  // Enable appending
    .create(true)  // Create the file if it doesn't exist
    .open("TTD.txt")?;
    let mut TTDwriter = BufWriter::new(TTDfile);  

    let number = 1000;

    let mut boundpercent:Vec<i32> = vec![0;RUNS];
    let mut targetpercent:Vec<i32> = vec![0;RUNS];
    let dna = placedna(50);
    let mut particles = Placeparticles(number, &mut dna.clone());
    let mut status:Vec<usize> = vec![0;number];
    Save(&dna,&particles,&mut PosWriter);
    for i in 0..RUNS{
        let (new_particles, new_status) = moveparticles(particles, &mut status, number, &dna);
        particles = new_particles;
        status = new_status.to_vec();
        status = detectcollision(&particles, status, &dna);
        boundpercent[i] = boundpercentfunction(&status);
        targetpercent[i] = targetpercentfunction(&status);
        if i%100 == 0{
            println!("Run {}/{}",i,RUNS);
            println!("Bound: {}/{}",boundpercent[i],number);
            println!("Target: {}/{}",targetpercent[i],number);
        }
    }
    saveboundpercent(targetpercent, &mut TTDwriter);
    Ok(())
}

fn placedna(dnanumber: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    // Function to generate DNA strands
    // Each strand is represented as a tuple of (origin, direction)
    // where origin is a point in 3D space and direction is a unit vector
    let mut dna: Vec<(Vec<f64>, Vec<f64>)> = Vec::new(); // Output vector
    let mut rng = rand::thread_rng(); // Random number generator
    use rand::Rng;
    use rand_distr::StandardNormal;

    /// Checks if an infinite line defined by `origin` and `direction` intersects the unit cube.
    /// Based on the "slab method" for line-box intersection.
    fn intersects_unit_box(origin: &[f64; 3], direction: &[f64; 3]) -> bool {
        let mut tmin = f64::NEG_INFINITY; // Minimum intersection parameter
        let mut tmax = f64::INFINITY;     // Maximum intersection parameter

        for i in 0..3 {
            if direction[i].abs() < 1e-8 {
                // Line is (almost) parallel to this axis.
                // If origin is outside the box slab on this axis, no intersection.
                if origin[i] < 0.0 || origin[i] > 1.0 {
                    return false;
                }
            } else {
                // Compute the intersection with the slabs at x=0 and x=1 (or y/z)
                let t1 = (0.0 - origin[i]) / direction[i];
                let t2 = (1.0 - origin[i]) / direction[i];
                // Swap if needed so t1 is near and t2 is far
                let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                // Update the interval
                tmin = tmin.max(t_near);
                tmax = tmax.min(t_far);
                // If the intervals do not overlap, the line misses the box
                if tmin > tmax {
                    return false;
                }
            }
        }

        true // The line intersects all three slabs — it hits the box
    }

    // Keep generating cylinders until we have the desired number
    while dna.len() < dnanumber {
        // --- Step 1: Sample a random unit direction vector using a Gaussian distribution ---
        let v1 = [
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
        ];
        // Normalize the vector to get a unit direction
        let norm = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
        let v1_norm = [v1[0] / norm, v1[1] / norm, v1[2] / norm];

        // --- Step 2: Generate an orthonormal basis using Gram-Schmidt ---
        // Create a second random vector (not collinear with v1_norm)
        let mut rand_vec = [
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
        ];
        // Project rand_vec onto v1_norm and subtract to make it orthogonal to v1_norm
        let dot = rand_vec[0]*v1_norm[0] + rand_vec[1]*v1_norm[1] + rand_vec[2]*v1_norm[2];
        for j in 0..3 {
            rand_vec[j] -= dot * v1_norm[j];
        }
        // Normalize the orthogonal vector
        let norm2 = (rand_vec[0].powi(2) + rand_vec[1].powi(2) + rand_vec[2].powi(2)).sqrt();
        let v2_norm = [rand_vec[0] / norm2, rand_vec[1] / norm2, rand_vec[2] / norm2];

        // --- Step 3: Compute third orthonormal vector using cross product ---
        let v3 = [
            v1_norm[1]*v2_norm[2] - v1_norm[2]*v2_norm[1],
            v1_norm[2]*v2_norm[0] - v1_norm[0]*v2_norm[2],
            v1_norm[0]*v2_norm[1] - v1_norm[1]*v2_norm[0],
        ];

        // --- Step 4: Generate a random point in the plane perpendicular to v1_norm ---
        // The point is defined as a linear combination of v2 and v3
        let r1: f64 = rng.gen_range(-1.0..2.0); // Coefficient for v2_norm
        let r2: f64 = rng.gen_range(-1.0..2.0); // Coefficient for v3
        let mut center = [0.0; 3];
        for j in 0..3 {
            // center = r1 * v2 + r2 * v3
            center[j] = r1 * v2_norm[j] + r2 * v3[j];
        }

        // --- Step 5: Check if the infinite line intersects the unit cube ---
        if intersects_unit_box(&center, &v1_norm) {
            // If valid, store as (origin, direction)
            dna.push((center.to_vec(), v1_norm.to_vec()));
        }
        // Else: discard and repeat
    }

    dna
}


fn Placeparticles(number: usize, dna: &Vec<(Vec<f64>, Vec<f64>)>) -> Vec<Vec<f64>> {
    let mut rng = rand::rng();
    let mut particles: Vec<Vec<f64>> = vec![vec![0.0; 3]; number];

    for i in 0..number {
        'placement: loop {
            // Generate a random point in the unit cube
            let p = vec![
            rng.random::<f64>(),
            rng.random::<f64>(),
            rng.random::<f64>(),
            ];

            // Check distance to all DNA strands
            for (point_on_line, direction) in dna.iter() {
                // Compute vector from point on the line to the particle
                let mut v = vec![0.0; 3];
                for j in 0..3 {
                    v[j] = p[j] - point_on_line[j];
                }

                // Project v onto the direction vector: t = (v • d) / |d|^2
                let dot_v_d = v.iter().zip(direction).map(|(vi, di)| vi * di).sum::<f64>();
                let d_norm_sq = direction.iter().map(|x| x * x).sum::<f64>();
                let t = dot_v_d / d_norm_sq;

                // Compute closest point on the line: point_on_line + t * direction
                let closest_point: Vec<f64> = point_on_line
                    .iter()
                    .zip(direction)
                    .map(|(pi, di)| pi + t * di)
                    .collect();

                // Compute squared distance from particle to the closest point on the line
                let dist_sq = p.iter()
                    .zip(closest_point.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();

                // If too close to any DNA strand, reject and retry
                if dist_sq < RSQRD {
                    continue 'placement;
                }
            }

            // Accept placement
            particles[i] = p;
            break 'placement;
        }
    }

    particles
}


fn moveparticles<'a>(mut particles:Vec<Vec<f64>>,  status: &'a mut Vec<usize>, number:usize, dna:&'a Vec<(Vec<f64>, Vec<f64>)>)-> (Vec<Vec<f64>>, &'a mut Vec<usize>){
    let mut rng = rand::rng();
    for i in 0..number{
        if status[i] == 0 {
            particles[i][0] += rng.sample::<f64, _>(rand_distr::StandardNormal) / 100.0;
            particles[i][1] += rng.sample::<f64, _>(rand_distr::StandardNormal) / 100.0;
            particles[i][2] += rng.sample::<f64, _>(rand_distr::StandardNormal) / 100.0;
        }

        if status[i] != 0 && status[i] != 999999{
            let DNAdirection = dna[status[i]-1].1.clone();
            let DNAcenter = dna[status[i]-1].0.clone();
            // Take the distance from the center of the DNA to the particle
        if (particles[i][0] - DNAcenter[0]).powi(2) + (particles[i][1] - DNAcenter[1]).powi(2) + (particles[i][2] - DNAcenter[2]).powi(2) > TARGET_RADIUS_SQRD || status[i] != 1{
                particles[i][0] += DNAdirection[0] * rng.sample::<f64, _>(rand_distr::StandardNormal) / 100.0;
                particles[i][1] += DNAdirection[1] * rng.sample::<f64, _>(rand_distr::StandardNormal) / 100.0;
                particles[i][2] += DNAdirection[2] * rng.sample::<f64, _>(rand_distr::StandardNormal) / 100.0;
                // Random chance to unbind
                let random_number: f64 = rng.r#gen::<f64>();
                if random_number < UNBINDCHANCE {
                    status[i] = 0; // Unbind the particle
                    // Generate a new random orthogonal vector to the DNA direction
                    let mut orthogonal_vector = vec![0.0; 3];

                    
                }
            }
        else{
            status[i] = 999999;
        }
    }
        
        if particles[i][0] > 1.0{
            particles[i][0] = 1.0;
        }
        if particles[i][0] < 0.0{
            particles[i][0] = 0.0;
        }
        if particles[i][1] > 1.0{
            particles[i][1] = 1.0;
        }
        if particles[i][1] < 0.0{
            particles[i][1] = 0.0;
        }
        if particles[i][2] > 1.0{
            particles[i][2] = 1.0;
        }
        if particles[i][2] < 0.0{
            particles[i][2] = 0.0;
        }
    }
    (particles, status)
}


fn detectcollision(
    particles: &Vec<Vec<f64>>, 
    mut status: Vec<usize>, 
    dna: &Vec<(Vec<f64>, Vec<f64>)>
) -> Vec<usize> {
    const BINDING_DISTANCE: f64 = 0.05;
    const BINDING_DISTANCE_SQRD: f64 = BINDING_DISTANCE * BINDING_DISTANCE;
    const BINDINGCHANCE: f64 = 0.1; // Adjust this to control P_on

    for (particle_index, particle) in particles.iter().enumerate() {
        if status[particle_index] != 0 {
            continue; // Skip already bound particles
        }

        let mut closest_strand: Option<usize> = None;
        let mut min_dist_sqrd = f64::MAX;

        // Find the closest strand within the binding threshold
        for (strand_index, (point, direction)) in dna.iter().enumerate() {
            // Vector from a point on the line to the particle
            let diff = [
                particle[0] - point[0],
                particle[1] - point[1],
                particle[2] - point[2],
            ];

            // Project this vector onto the line direction
            let dot = diff[0]*direction[0] + diff[1]*direction[1] + diff[2]*direction[2];

            // Closest point on the infinite line
            let closest = [
                point[0] + dot * direction[0],
                point[1] + dot * direction[1],
                point[2] + dot * direction[2],
            ];

            // Distance^2 between particle and closest point
            let dist_sqrd = 
                (particle[0] - closest[0]).powi(2) +
                (particle[1] - closest[1]).powi(2) +
                (particle[2] - closest[2]).powi(2);

            if dist_sqrd < BINDING_DISTANCE_SQRD && dist_sqrd < min_dist_sqrd {
                min_dist_sqrd = dist_sqrd;
                closest_strand = Some(strand_index);
            }
        }

        // Attempt to bind to the closest valid strand
        if let Some(strand_index) = closest_strand {
            let random_number: f64 = rand::random();
            if random_number < BINDINGCHANCE {
                status[particle_index] = (strand_index + 1); // Store index+1
            }
        }
    }

    status
}


fn boundpercentfunction(status:&Vec<usize>)->i32{
    let mut amount:i32 = 0;
    for element in status{
        if *element != 0{
            amount+=1;
        }
    }
    return amount
}

fn targetpercentfunction(status:&Vec<usize>)->i32{
    let mut amount:i32 = 0;
    for element in status{
        if *element == 999999{
            amount+=1;
        }
    }
    return amount
}

fn testboundpercent(status:&Vec<usize>, number:usize)->String{
    let mut amount:i32 = 0;
    for element in status{
        if *element != 0{
            amount+=1;
        }
    }
    let string_a:String = amount.to_string();
    let string_b:String = number.to_string();
    return string_a + " / " + &string_b;
}

fn Save(
    dna: &Vec<(Vec<f64>, Vec<f64>)>,
    particles: &Vec<Vec<f64>>,
    pos_writer: &mut BufWriter<File>
) -> Result<(), Box<dyn std::error::Error>> {
        // First line of output.
        writeln!(pos_writer, "DNA")?;
        for (point, direction) in dna.iter() {
            writeln!(
                pos_writer,
                "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                point[0], point[1], point[2],
                direction[0], direction[1], direction[2]
            )?;
        }

        writeln!(pos_writer, "Particles start")?;
        for triplet in particles.iter() {
            writeln!(pos_writer, "{:.4},{:.4},{:.4}", triplet[0], triplet[1], triplet[2])?;
        }

    pos_writer.flush()?;
    Ok(())
}

fn saveboundpercent(boundpercent:Vec<i32>, TTDPosWriter: &mut BufWriter<File>)-> Result<(), Box<dyn std::error::Error>>{
    for i in 0..boundpercent.len(){
        write!(TTDPosWriter, "{},",boundpercent[i])?;
    }
    writeln!(TTDPosWriter, "")?;
    TTDPosWriter.flush()?;
    Ok(())

}
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a dummy DNA strand represented as (point, direction).
    fn dummy_dna_strand(point: [f64; 3]) -> (Vec<f64>, Vec<f64>) {
        (vec![point[0], point[1], point[2]], vec![0.0, 0.0, 1.0]) // Z-axis direction
    }

    #[test]
    fn test_detectcollision_binds_nearby_particle() {
        let particles = vec![vec![0.5, 0.5, 0.5], vec![0.2, 0.2, 0.2]];
        let status = vec![0, 0];
        let dna = vec![dummy_dna_strand([0.5, 0.5, 0.0])];

        let mut at_least_one_bound = false;
        for _ in 0..100 {
            let updated_status = detectcollision(&particles, status.clone(), &dna);
            if updated_status.iter().any(|&s| s != 0) {
                at_least_one_bound = true;
                break;
            }
        }

        assert!(at_least_one_bound, "Expected at least one particle to bind after multiple tries");
    }

    #[test]
    fn test_detectcollision_no_binding_if_far() {
        let particles = vec![vec![0.5, 0.5, 0.5], vec![0.2, 0.2, 0.2]];
        let status = vec![0, 0];
        let dna = vec![dummy_dna_strand([5.0, 5.0, 5.0])]; // Far away DNA

        let updated_status = detectcollision(&particles, status.clone(), &dna);

        assert!(updated_status.iter().all(|&s| s == 0), "Expected no binding for far DNA");
    }

    #[test]
    fn test_detectcollision_multiple_dna_strands_binding_preference() {
        let particles = vec![vec![0.51, 0.5, 0.5], vec![0.19, 0.2, 0.2]];
        let status = vec![0, 0];
        let dna = vec![
            dummy_dna_strand([0.5, 0.5, 0.0]),
            dummy_dna_strand([0.2, 0.2, 0.0])
        ];

        // Run many times to ensure both bind if within distance
        let mut updated_status = status.clone();
        for _ in 0..100 {
            updated_status = detectcollision(&particles, status.clone(), &dna);
            if updated_status[0] != 0 && updated_status[1] != 0 {
                break;
            }
        }

        assert_eq!(updated_status[0], 1, "Particle 0 should bind to DNA strand 0");
        assert_eq!(updated_status[1], 2, "Particle 1 should bind to DNA strand 1");
    }
}