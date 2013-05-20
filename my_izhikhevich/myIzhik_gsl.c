/**
 * my C implementation of Eugene M. Izhikevich's neuron model
 *
 * this version uses the Gnu Scientific Library (GSL)
 */

#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/*int main ( int argc, char **argv) { */
int main ( void ) {

	/* useful generic counters */
	unsigned int i, j;

	/* stream container of the output */
	FILE *out;
	char out_prefix[] = "firings.dat";
	int out_fd = mkstemp(out_prefix);
	if ( out_fd < 0 )
	{
		perror("Can't create temp file, trying overwriting");
		out = fopen(out_prefix, "w+");
	}
	else
		out = fdopen(out_fd, "w+");
	
	if ( out == NULL )
	{
		perror("Can't open file, bailing out");
		return -1;
	}
	char outstring[128];

	/* allocate and init the default random number generator */
	gsl_rng *rng;
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	rng = gsl_rng_alloc(T);

	const size_t Ne = 8000;
	const size_t Ni = 2000;
	const size_t N = Ne + Ni;

	/* time constant and total simulation time (1sec) */
	unsigned int t;
	const unsigned int SIM_TIME = 1000;

	/* izhikevic's parameters */

	gsl_vector 
		/* the input vector */
		*I = gsl_vector_alloc(N),
		/* parameters of Izhikevich's model */
		*u = gsl_vector_alloc(N),
		*v = gsl_vector_alloc(N),
		/* parameters for different types of neurones */
		*izhik_a = gsl_vector_alloc(N),
		*izhik_b = gsl_vector_alloc(N),
		*izhik_c = gsl_vector_alloc(N),
		*izhik_d = gsl_vector_alloc(N),
		/* temporary vectors */
		*tmp1 = gsl_vector_alloc(N),
		*tmp2 = gsl_vector_alloc(N);

	/* used to compute input from firing neurones */
	gsl_vector_view this_col;

	/* 
	 * adjust 
	 * for excitatory neurones:
	 * (c_i, d_i) = ( -65, 8 ) + ( 15, -6 ) r_i^2
	 * for inh:
	 * (a_i, b_i) = ( 0.02, 0.25) + ( 0.08, -0.05) r_i
	 * where 
	 * r_i is a random variable uniformly distributed in [0,1]
	 */
	double r_i;

	for ( i = 0 ; i < Ne ; i++ )
	{
		gsl_vector_set(izhik_a, i, 0.02);
		gsl_vector_set(izhik_b, i, 0.2);
		r_i = gsl_ran_flat(rng, 0.0, 1.0);
		gsl_vector_set(izhik_c, i, -65.0 + 15.0 * r_i * r_i);
		r_i = gsl_ran_flat(rng, 0.0, 1.0);
		gsl_vector_set(izhik_d, i, 8.0 - 6.0 * r_i * r_i);
	}
	for ( i = Ne ; i < N ; i++ ) 
	{
		r_i = gsl_ran_flat(rng, 0.0, 1.0);
		gsl_vector_set(izhik_a, i, 0.02 + 0.08 * r_i);
		r_i = gsl_ran_flat(rng, 0.0, 1.0);
		gsl_vector_set(izhik_b, i, 0.25 - 0.05 * r_i);
		gsl_vector_set(izhik_c, i, -65.0);
		gsl_vector_set(izhik_d, i, 2.0);
	}


	/* set initial values of v and u */
	gsl_vector_set_all(v, -65.0);
	gsl_vector_memcpy(tmp1, izhik_b);
	gsl_vector_mul(tmp1, v);
	gsl_vector_memcpy(u, tmp1);


	/* the connectivity matrix */
	gsl_matrix *S = gsl_matrix_alloc(N, N); 
	for ( i = 0 ; i < N ; i++ ) 
	{
		for ( j = 0 ; j < Ne ; j++ ) 
			gsl_matrix_set(S, i, j, gsl_ran_flat(rng, 0.0, 0.5));
		for ( j = Ne ; j < N ; j++ )
			gsl_matrix_set(S, i, j, gsl_ran_flat(rng, -1.0, 0.0));
	}

	/* contains the indices of neurones which fired */
	unsigned int fired[N];
	/* the total number of neurons which fired */
	size_t num_fired;
	
	/* temporary variable */
	double new_u; 

	for ( t = 1 ; t <= SIM_TIME ; t++ )
	{
		/* initial value of thalamic input */
		for ( i = 0 ; i < Ne ; i++ )
			gsl_vector_set(I, i, gsl_ran_flat(rng, 0.0, 5.0));
		for ( i = Ne ; i < N ; i++ )
			gsl_vector_set(I, i, gsl_ran_flat(rng, 0.0, 2.0));

		/* find the neurones which fired (v >= 30 ) */
		for ( i = 0, j = 0 ; i < N ; i++ )
			if ( gsl_vector_get(v, i) >= 30.0 )
				fired[j++] = i;

		/* used to print to buffer */
		num_fired = j;

		/* 
		 * after-spike resetting 
		 * 
		 * in LaTeX:
		 * if v \geq 30 mV, then (group) v \larrow c \\ u \larrow u + d
		 * */
		for ( i = 0 ; i < num_fired ; i++ )
		{
			gsl_vector_set(v, fired[i], gsl_vector_get(izhik_c, fired[i]));
			new_u = gsl_vector_get(u, fired[i]) + gsl_vector_get(izhik_d, fired[i]);
			gsl_vector_set(u, fired[i], new_u);

			/* record to stream the firings neurones */
			sprintf(outstring, "%d %d\n", t, fired[i]);
			if ( (fputs(outstring, out)) == EOF )
				perror("fputs fail");

			/* 
			 * I = I + sum ( S(:, fired), 2); 
			 * a.k.a.
			 * from the matrix S
			 * extract the column of the neurones that fired
			 * and sum the rows
			 * obtain a vector of same size of I and add the two
			 */
			this_col = gsl_matrix_column(S, fired[i]);
			 gsl_vector_add(I, &this_col.vector);
		}

		/* 
		 * recalculate potentials:
		 *
		 * in MATLAB:
		 * v = v + 0.5 * (0.04 * v.^2 + 5 * v +140 - u+I);
		 * v = v + 0.5 * (0.04 * v.^2 + 5 * v +140 - u+I);
		 * u = u + a. * (b.* v-u);
		 *
		 * in LaTeX:
		 * v' = 0.04 v^2 + 5 v + 140 - u + I
		 * u' = a(bv - u)
		 */
		 gsl_vector_memcpy(tmp1, v);
		 gsl_vector_mul(tmp1, v);	/* v.^2 */
		 gsl_vector_scale(v, 0.04); /* 0.04 * v.^2 */
		 gsl_vector_memcpy(tmp2, v);
		 gsl_vector_scale(tmp2, 5.0);	/* 5 * v */
		 gsl_vector_add(tmp1, tmp2);	/* 0.04 * v.^ 2 + 5 * v */
		 gsl_vector_add_constant(tmp1, 140.0);
		 gsl_vector_sub(tmp1, u);
		 gsl_vector_add(tmp1, I);
		 gsl_vector_scale(tmp1, 0.5);
		 gsl_vector_add(v, tmp1);

		 gsl_vector_memcpy(tmp1, v);
		 gsl_vector_mul(tmp1, v);	/* v.^2 */
		 gsl_vector_scale(v, 0.04); /* 0.04 * v.^2 */
		 gsl_vector_memcpy(tmp2, v);
		 gsl_vector_scale(tmp2, 5.0);	/* 5 * v */
		 gsl_vector_add(tmp1, tmp2);	/* 0.04 * v.^ 2 + 5 * v */
		 gsl_vector_add_constant(tmp1, 140.0);
		 gsl_vector_sub(tmp1, u);
		 gsl_vector_add(tmp1, I);
		 gsl_vector_scale(tmp1, 0.5);
		 gsl_vector_add(v, tmp1);

		 gsl_vector_memcpy(tmp1, v);
		 gsl_vector_mul(tmp1, izhik_b);
		 gsl_vector_sub(tmp1, u);
		 gsl_vector_mul(tmp1, izhik_a);
		 gsl_vector_add(u, tmp1);
	}

	/* don't forget to keep flushing */
	fflush(out);

	/* free stuff */
	gsl_rng_free(rng);
	gsl_vector_free(I);
	gsl_vector_free(u);
	gsl_vector_free(v);
	gsl_matrix_free(S);
	fclose(out);
	return 0;

}
