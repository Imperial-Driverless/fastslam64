#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265359
#endif


#define MIN(a,b) (((a)<(b))?(a):(b))

__device__ double* get_particle(double *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

// Manual extern "C" to stop name mangling shenanigans
// Otherwise doesn't compile because curand complains
extern "C" {

__global__ void reset(int *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d[i] = n;
}

__global__ void prepermute(int *ancestors, int *d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // d[ancestors[i]] = i;

    int *p = d + ancestors[i];
    atomicMin(p, i);
}

__global__ void permute(int *ancestors, int *c, int *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int x = d[ancestors[i]];
    if(x != i) {
        x = i;
        while(d[x] < n) {
            x = d[x];
        }
        d[x] = i;
    }
}


__global__ void write_to_c(int *ancestors, int *c, int *d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    c[i] = ancestors[d[i]];
}


// Based on https://stackoverflow.com/questions/46169633/how-to-generate-random-number-inside-pycuda-kernel    
// Each thread has a random state
__device__ curandState_t* states[N_PARTICLES];


// This function is only called once to initialize the rngs.
__global__ void init_rng(int seed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t* s = new curandState_t;
    curand_init(seed, i, 0, s);
    states[i] = s;
}


__global__ void predict_from_imu(double *particles,
    double x, double y, double theta, double sigma_x, double sigma_y, double sigma_theta) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double *particle = get_particle(particles, i);
    // curand_normal() samples from standard normal
    // to get a general N(mu, sigma), we use Y = mu + sigma*X,
    // though in our case mu=0.
    particle[0] = x + sigma_x * curand_normal(states[i]);
    particle[1] = y + sigma_y * curand_normal(states[i]);
    particle[2] = theta + sigma_theta * curand_normal(states[i]);
    particle[2] = atan2(sin(particle[2]), cos(particle[2]));
}

// Moves particles based on the control input and movement model.
__global__ void predict_from_model(double *particles, double ua, double ub, double sigma_a, double sigma_b, double dt) {
    if(ua == 0.0 && ub == 0.0) {
        return;
    }

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double *particle = get_particle(particles, i);

    ua += sigma_a * curand_normal(states[i]);
    ub += sigma_b * curand_normal(states[i]);

    double angle = particle[2];

    // curand_normal() samples from standard normal
    // to get a general N(mu, sigma), we use Y = mu + sigma*X,
    // though in our case mu=0.
    particle[2] += (ua * dt);
    // particle[2] = fmod(particle[2], (double)(2*M_PI));
    particle[2] = atan2(sin(particle[2]), cos(particle[2]));


    double dist = (ub * dt);
    particle[0] += cos(angle) * dist;
    particle[1] += sin(angle) * dist;
}

// Moves particles based on the control input and movement model.
__global__ void predict_from_fsonline_model(double *particles, double ua, double ub, double uc, double sigma_a, double sigma_b, double dt) {
    if(ua == 0.0 && ub == 0.0 && uc == 0.0) {
        return;
    }

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double *particle = get_particle(particles, i);

    ua += sigma_a * curand_normal(states[i]);
    ub += sigma_b * curand_normal(states[i]);
    uc += sigma_a * curand_normal(states[i]);

    // curand_normal() samples from standard normal
    // to get a general N(mu, sigma), we use Y = mu + sigma*X,
    // though in our case mu=0.
    particle[2] += (ua * dt);
    // particle[2] = fmod(particle[2], (double)(2*M_PI));
    particle[2] = fmod(particle[2], (double)(2*M_PI));
    // particle[2] = atan2(sin(particle[2]), cos(particle[2]));

    double dist = (ub * dt);
    particle[0] += cos(particle[2]) * dist;
    particle[1] += sin(particle[2]) * dist;

    particle[2] += (uc * dt);
    particle[2] = fmod(particle[2], (double)(2*M_PI));
}




/*
 * Copies particles in place given by the ancestor vector
 */
__global__ void copy_inplace(
    double *particles, int *ancestors)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i == ancestors[i]) {
        return;
    }

    double *source = get_particle(particles, ancestors[i]);
    double *dest = get_particle(particles, i);

    int max_landmarks = (int)source[4];
    int n_landmarks = (int)source[5];

    dest[0] = source[0];
    dest[1] = source[1];
    dest[2] = source[2];
    dest[3] = source[3];
    dest[4] = source[4];
    dest[5] = source[5];

    for(int k = 0; k < n_landmarks; k++) {
        dest[6+2*k] = source[6+2*k];
        dest[6+2*k+1] = source[6+2*k+1];

        dest[6+2*max_landmarks+4*k] = source[6+2*max_landmarks+4*k];
        dest[6+2*max_landmarks+4*k+1] = source[6+2*max_landmarks+4*k+1];
        dest[6+2*max_landmarks+4*k+2] = source[6+2*max_landmarks+4*k+2];
        dest[6+2*max_landmarks+4*k+3] = source[6+2*max_landmarks+4*k+3];

        dest[6+6*max_landmarks+k] = source[6+6*max_landmarks+k];
    }
}
}

__global__ void copy_inplace_coalesced(
    double *particles, int *ancestors)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    for(int k = 0; k < N_PARTICLES; k++) {
        double *source = get_particle(particles, ancestors[i]);
        double *dest = get_particle(particles, i);

        dest[i] = source[i];
    }
}

extern "C" {
__global__ void reset_weights(double *particles) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    double *particle = get_particle(particles, i);
    particle[3] = 1.0/N_PARTICLES;
}

__global__ void systematic_resample(double *weights, double *cumsum, double rand, int *ancestors) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    int left = ceil((cumsum[i]*N_PARTICLES - weights[i]*N_PARTICLES) - rand);
    int right = ceil((cumsum[i]*N_PARTICLES) - rand);

    for(int j = left; j < right; j++) {
        ancestors[j] = i;
    }
}
}


/*
 * Calculates neff.
 * Needs to run in a single block.
 */
 __global__ void get_neff(double *particles, double *neff) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    double square_sum = 0;

    for (int i = idx; i < N_PARTICLES; i += THREADS) {
        double *particle = get_particle(particles, i);
        square_sum += (double)particle[3] * (double)particle[3];
    }

    __shared__ double r_square_sum[THREADS];
    r_square_sum[idx] = square_sum;

    __syncthreads();

    for (int size = THREADS/2; size > 0; size /= 2) {
        if (idx < size) {
            r_square_sum[idx] += r_square_sum[idx + size];
        }
        __syncthreads();
    }

    if (idx == 0) {
        *neff = 1.0/r_square_sum[0];
    }
}

extern "C" {
/*
 * Sums particle weights.
 * Needs to run in a single block.
 */
__global__ void sum_weights(double *particles, double *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    for (int i = idx; i < N_PARTICLES; i += THREADS) {
        double *particle = get_particle(particles, i);
        sum += (double)particle[3];
    }

    __shared__ double r[THREADS];
    r[idx] = sum;
    __syncthreads();

    for (int size = THREADS/2; size > 0; size /= 2) {
        if (idx < size) {
            r[idx] += r[idx + size];
        }
        __syncthreads();
    }

    if (idx == 0) {
        *out = r[0];
    }
}

/*
 * Rescales particle weights so that \sum_i w_i = 1
 */
__global__ void divide_weights(double *particles, double *s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = s[0];
    double *particle = get_particle(particles, i);

    if(sum > 0) {
        particle[3] /= sum;
    } else {
        particle[3] = 1.0/N_PARTICLES;
    }
}
}


typedef struct 
{
    double (*measurements)[2];
    int n_measurements;
    double *measurement_cov;
} landmark_measurements;

__device__ double mod_angle(double angle) {
    return atan2(sin(angle), cos(angle));
}

__device__ double vecnorm(double *v) {
    return sqrt(v[0]*v[0] + v[1]*v[1]);
}

__device__ bool in_sensor_range(double *position, double *landmark, double range, double fov) {
    double x = position[0];
    double y = position[1];
    double theta = position[2];
    double lx = landmark[0];
    double ly = landmark[1];

    double va[] = {lx - x, ly - y};
    double vb[] = {range * cos(theta), range * sin(theta)};

    if(vecnorm(va) > range) {
        return false;
    }

    double angle = acos(
        (va[0]*vb[0] + va[1]*vb[1])/(vecnorm(va)*vecnorm(vb))
    );

    if(angle <= (fov/2)) {
        return true;
    } else {
        return false;
    }
}

__device__ void to_coords(double *particle, double *in, double *out) {
    double x = particle[0];
    double y = particle[1];
    double theta = particle[2];

    double range = in[0];
    double bearing = in[1];

    out[0] = x + range * cos(bearing + theta);
    out[1] = y + range * sin(bearing + theta);
}

__device__ double* get_mean(double *particle, int i)
{
    return (particle + 6 + 2*i);
}

__device__ double* get_cov(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    return (particle + 6 + 2*max_landmarks + 4*i);
}

__device__ double* get_landmark_prob(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    return (particle + 6 + 6*max_landmarks + i);
}

__device__ void increment_landmark_prob(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    double *prob = (particle + 6 + 6*max_landmarks + i);
    prob[0] += 1.0;
}

__device__ void decrement_landmark_prob(double *particle, int i)
{
    int max_landmarks = (int)particle[4];
    double *prob = (particle + 6 + 6*max_landmarks + i);
    prob[0] -= 1.0;
}

__device__ int get_n_landmarks(double *particle)
{
    return (int)particle[5];
}

__device__ void add_landmark(double *particle, double mean[2], double *cov)
{
    int n_landmarks = (int)particle[5];
    particle[5] = (double)(n_landmarks + 1);

    double *new_mean = get_mean(particle, n_landmarks);
    double *new_cov = get_cov(particle, n_landmarks);
    double *new_prob = get_landmark_prob(particle, n_landmarks);

    new_mean[0] = mean[0];
    new_mean[1] = mean[1];

    new_cov[0] = cov[0];
    new_cov[1] = cov[1];
    new_cov[2] = cov[2];
    new_cov[3] = cov[3];

    new_prob[0] = 1.0;
}

__device__ void remove_landmark(double *particle, int i)
{
    int n_landmarks = (int)particle[5];

    double *mean_a = get_mean(particle, i);
    double *mean_b = get_mean(particle, n_landmarks - 1);

    mean_a[0] = mean_b[0];
    mean_a[1] = mean_b[1];

    double *cov_a = get_cov(particle, i);
    double *cov_b = get_cov(particle, n_landmarks - 1);

    cov_a[0] = cov_b[0];
    cov_a[1] = cov_b[1];
    cov_a[2] = cov_b[2];
    cov_a[3] = cov_b[3];

    double *prob_a = get_landmark_prob(particle, i);
    double *prob_b = get_landmark_prob(particle, n_landmarks - 1);

    prob_a[0] = prob_b[0];
    
    particle[5] = (double)(n_landmarks - 1);
}

__device__ void vecmul(double *A, double *u, double *v)
{
    double a = A[0];
    double b = A[1];
    double c = A[2];
    double d = A[3];

    double e = u[0];
    double f = v[1];

    v[0] = a*e + b*f;
    v[1] = c*e + d*f;
}

__device__ void matmul(double *A, double *B, double *C)
{
    double a = A[0];
    double b = A[1];
    double c = A[2];
    double d = A[3];

    double e = B[0];
    double f = B[1];
    double g = B[2];
    double h = B[3];

    C[0] = a*e + b*g;
    C[1] = a*f + b*h;
    C[2] = c*e + d*g;
    C[3] = c*f + d*h;
}

__device__ void matmul_jacobian(double *H, double *E, double *R, double *S)
{
    double a = H[0];
    double b = H[1];
    double c = H[2];
    double d = H[3];

    double Ht[] = {
        a, c,
        b, d
    };

    matmul(H, E, S);
    matmul(S, Ht, S);

    S[0] += R[0];
    S[1] += R[1];
    S[2] += R[2];
    S[3] += R[3];
}

__device__ void pinv(double *A, double *B)
{
    double a = A[0];
    double b = A[1];
    double c = A[2];
    double d = A[3];

    double scalar = 1/(a*d - b*c);

    B[0] = scalar * d;
    B[1] = scalar * (-b);
    B[2] = scalar * (-c);
    B[3] = scalar * a;
}

__device__ double pdf(double *x, double *mean, double* cov)
{
    double cov_inv[] = {0, 0, 0, 0};
    pinv(cov, cov_inv);

    double scalar = 1/(2*M_PI*sqrt(cov[0]*cov[3] - cov[1]*cov[2]));

    double m = x[0] - mean[0];
    double n = mod_angle(x[1] - mean[1]);

    double arg = m*m*(cov_inv[0]) + n*n*(cov_inv[3]) + m*n*(cov_inv[1] + cov_inv[2]);

    return scalar * exp(-0.5 * arg);
}

__device__ void add_measurement_as_landmark(double *particle, double *measurement, double *measurement_cov)
{
    double pos[] = { particle[0], particle[1] };
    double landmark[] = {0, 0};
    to_coords(particle, measurement, landmark);

    double q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);

    double H[] = {
        (landmark[0] - pos[0])/(sqrt(q)), (landmark[1] - pos[1])/(sqrt(q)),
        -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
    };

    pinv(H, H);

    double H_inv_t[] = {
        H[0], H[2],
        H[1], H[3]
    };

    double S[] = {
        0, 0, 0, 0
    };

    matmul(H, measurement_cov, S);
    matmul(S, H_inv_t, S);
    add_landmark(particle, landmark, S);
}


__device__ void add_measurements_as_landmarks(double *particle, landmark_measurements *measurements)
{
    int n_measurements = measurements->n_measurements;
    double *measurement_cov = measurements->measurement_cov;

    for(int i = 0; i < n_measurements; i++) {
        add_measurement_as_landmark(particle, measurements->measurements[i], measurement_cov);
    }
}


__device__ double compute_dist(double *particle, int i, double *measurement, double *measurement_cov)
{
    double *landmark = get_mean(particle, i);
    double measurement_xy[] = {0, 0};
    to_coords(particle, measurement, measurement_xy);

    double dist = sqrt(
        (landmark[0] - measurement_xy[0])*(landmark[0] - measurement_xy[0]) +
        (landmark[1] - measurement_xy[1])*(landmark[1] - measurement_xy[1])
    );
    
    return dist;
}


__device__ void update_landmarks(int id, double *particle, landmark_measurements *measurements, int *in_range, int *n_matches, double range, double fov, double thresh)
{
    double *measurement_cov = measurements->measurement_cov;
    int n_measurements = measurements->n_measurements;

    int n_landmarks = get_n_landmarks(particle);

    int n_in_range = 0;
    for(int i = 0; i < n_landmarks; i++) {
        n_matches[i] = 0;
        double *mean = get_mean(particle, i);
        in_range[n_in_range] = i;
        n_in_range++;
        // if(in_sensor_range(particle, mean, range, fov)) {
        //     in_range[n_in_range] = i;
        //     n_in_range++;
        // }
    }

    for(int i = 0; i < n_measurements; i++) {
        double best = 1000000.0;
        int best_idx = -1;

        for(int j = 0; j < n_in_range; j++) {
            double dist = compute_dist(particle, in_range[j], measurements->measurements[i], measurement_cov);

            if(dist <= thresh && dist < best) {
                best = dist;
                best_idx = in_range[j];
            }
        }

        if(best_idx != -1) {
            n_matches[best_idx]++;
        }

        if(best_idx != -1) {
            double *landmark = get_mean(particle, best_idx);
            double pos[] = { particle[0], particle[1] };
            double theta = particle[2];

            double q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);
            double measurement_predicted[] = {
                sqrt(q), mod_angle(atan2(landmark[1] - pos[1], landmark[0] - pos[0]) - theta)
            };

            double residual[2] = {
                measurements->measurements[i][0] - measurement_predicted[0],
                mod_angle(measurements->measurements[i][1] - measurement_predicted[1])
            };


            double H[] = {
                (landmark[0] - pos[0])/(sqrt(q)), (landmark[1] - pos[1])/(sqrt(q)),
                -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
            };

            double Ht[] = {
                H[0], H[2],
                H[1], H[3]
            };

            double S[] = {
                0, 0, 0, 0
            };

            double *landmark_cov = get_cov(particle, best_idx);
        
            matmul_jacobian(H, landmark_cov, measurement_cov, S);
            double S_inv[] = {0, 0, 0, 0};
            pinv(S, S_inv);


            double Q[] = {0, 0, 0, 0};
            double K[] = { 0, 0, 0, 0 };
            matmul(landmark_cov, Ht, Q);
            matmul(Q, S_inv, K);

            double K_residual[] = { 0, 0 };
            vecmul(K, residual, K_residual);
            landmark[0] += K_residual[0];
            landmark[1] += K_residual[1];

            double KH[] = { 0, 0, 0, 0};
            matmul(K, H, KH);
            double new_cov[] = { 1 - KH[0], -KH[1], -KH[2], 1 - KH[3] };
            matmul(new_cov, landmark_cov, new_cov);
            landmark_cov[0] = new_cov[0];
            landmark_cov[1] = new_cov[1];
            landmark_cov[2] = new_cov[2];
            landmark_cov[3] = new_cov[3];

            particle[3] *= pdf(measurements->measurements[i], measurement_predicted, S);

            increment_landmark_prob(particle, best_idx);

        } else {
            add_measurement_as_landmark(particle, measurements->measurements[i], measurement_cov);
        }
    }

    // for(int i = n_in_range - 1; i > 0; i--) {
    //     int idx = in_range[i];
    //     if(n_matches[idx] == 0) {
    //         decrement_landmark_prob(particle, idx);
    //         double prob = get_landmark_prob(particle, idx)[0];
    //         if(prob < 0) {
    //             remove_landmark(particle, idx);
    //         }
    //     } 
    // }
}

extern "C" {
__global__ void update(
    double *particles, int block_size, int *scratchpad_mem, int scratchpad_size, double measurements_array[][2], int n_particles, int n_measurements,
    double *measurement_cov, double threshold, double range, double fov, int max_landmarks)
{

    if(n_measurements == 0) {
        return;
    }

    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int *scratchpad = scratchpad_mem + (2 * thread_id * max_landmarks);
    int *in_range = scratchpad;
    int *n_matches = in_range + max_landmarks;

    landmark_measurements measurements;
    measurements.n_measurements = n_measurements;
    measurements.measurement_cov = measurement_cov;
    measurements.measurements = measurements_array;

    for(int k = 0; k < block_size; k++) {
        int particle_id = thread_id*block_size + k;
        if(particle_id >= n_particles) {
            return;
        }
        
        double *particle = get_particle(particles, particle_id);
        int n_landmarks = get_n_landmarks(particle);
    
        if(n_landmarks == 0) {
            add_measurements_as_landmarks(particle, &measurements);
            continue;
        }

        update_landmarks(particle_id, particle, &measurements, in_range, n_matches, range, fov, threshold);
    }
}

/*
 * Extracts weights from particles.
 */
__global__ void get_weights(double *particles, double *weights) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double *particle = get_particle(particles, i);
    weights[i] = (double)particle[3];
}

/*
 * Calculates the mean position of all particles.
 * Needs to run in a single block.
 */
__global__ void get_mean_position(double *particles, double *mean) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    double x = 0;
    double y = 0;
    double theta = 0;

    for (int i = idx; i < N_PARTICLES; i += THREADS) {
        double *particle = get_particle(particles, i);
        x += particle[3] * particle[0];
        y += particle[3] * particle[1];
        theta += particle[3] * particle[2];
    }

    __shared__ double r_x[THREADS];
    __shared__ double r_y[THREADS];
    __shared__ double r_theta[THREADS];

    r_x[idx] = x;
    r_y[idx] = y;
    r_theta[idx] = theta;

    __syncthreads();

    for (int size = THREADS/2; size>0; size/=2) {
        if (idx<size) {
            r_x[idx] += r_x[idx+size];
            r_y[idx] += r_y[idx+size];
            r_theta[idx] += r_theta[idx+size];
        }
        __syncthreads();
    }

    if (idx == 0) {
        mean[0] = r_x[0];
        mean[1] = r_y[0];
        mean[2] = r_theta[0];
    }
}

}
