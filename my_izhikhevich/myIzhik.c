/**
 * my C implementation of Eugene M. Izhikevich's neuron model
 *
 * lorenzo.grespan@gmail.com
 */

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * if want to use GSL (there's no GSL support on cluster..)
 *
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
 *
 * also VERY IMPORTANT don't forget the seeding of the RNG below
 */

/* in case you want to change RNG, do it here 
 * (and don't forget the initialisation, below)
 *
 * example below is to use GSL
 *
#define genrand() gsl_ran_flat(rng, 0.0, 1.0)
#define genrandn() gsl_ran_gaussian(rng, 1.0)
 */
#define genrand() drand48()
#define genrandn() gaus()

#define DEFAULT_RANDSEED 42
#define DEFAULT_SIMTIME 1000
#define DEFAULT_OUTPREFIX "firings_XXXXXX"
#define DEFAULT_NE 800
#define DEFAULT_NI 200

static void usage(char *progname) {
	printf("Usage: %s\n", progname);
		printf("options:\n-s <random seed> \n-t <simulation time> \n-f <output file> \n-e <excitatory neurones> \n-i <inhibitory neurones> \n-h (this help)\n");
	printf("\n");
}

static void print_runningparams(long int seed, long int simtime, int ne, int ni, char* outfile) {
#ifdef _OPENMP
	printf("Using OpenMP\n");
#else
	printf("Not using OpenMP\n");
#endif
	printf("random seed: %ld\n", seed);
	printf("simulation time: %ldms\n", simtime);
	printf("excitatory neurones: %d\n", ne);
	printf("inhibitory neurones: %d\n", ni);
	printf("\nsaving data in: %s\n", outfile);
	printf("\n");
}

/**
 * Generation of gaussian deviates using Box-Muller transformation
 *
 * following code seen on
 * http://www.taygeta.com/random/gaussian.html
 * and on
 * 'numerical recipes in C', 3rd edition
 *
 * TODO: check copyright..?
 */
static double gaus(void) {
	static int iset = 0;
	static double gset;
	double fac, rsq, v1, v2;

	if ( iset == 0 )
	{
		do {
			v1 = 2.0 * drand48() - 1.0;
			v2 = 2.0 * drand48() - 1.0;
			rsq = v1 * v1 + v2 * v2;
		} while ( rsq >= 1.0 || rsq == 0.0 );
		fac = sqrt(-2.0 *log(rsq)/rsq);

		gset = v1*fac;
		iset = 1;
		return v2*fac;
	}
	else
	{
		iset = 0;
		return gset;
	}
}

int main ( int argc, char **argv) { 

	long int seed, simtime;
	seed = DEFAULT_RANDSEED;
	simtime = DEFAULT_SIMTIME;

	/* number of neurones */
	int Ne = DEFAULT_NE;
	int Ni = DEFAULT_NI;

	if ( argc == 1 )
		usage(argv[0]);
	/* next few lines 'borrowed' from getopt(3) manpage */
	int usefile = 0, ch;
	char* outfile = NULL;
	while ((ch = getopt(argc, argv, "s:t:f:e:i:h")) != -1) {
		switch (ch) {
			case 's':
				seed = strtol(optarg, NULL, 10);
				break;
			case 't':
				simtime = strtol(optarg, NULL, 10);
				break;
			case 'f':
				usefile = 1;
				outfile = malloc(strlen(optarg));
				outfile = strncpy(outfile, optarg, strlen(optarg));
				break;
			case 'e':
				Ne = atoi(optarg);
				break;
			case 'i':
				Ni = atoi(optarg);
				break;
			case '?':
				break;
			case 'h':
				usage(argv[0]);
				exit(-1);
			default:
				usage(argv[0]);
		}
	}
	argc -= optind;
	argv += optind;

	/* useful generic counters */
	int i, j;

	/* stream container of the output */
	FILE *out;
	if ( usefile == 1 )
	{
		out = fopen(outfile, "w+");
	}
	else
	{
		char out_prefix[] = DEFAULT_OUTPREFIX;
		int out_fd = mkstemp(out_prefix);
		if ( out_fd < 0 )
		{
			perror("Can't create random temp file, bailing out");
			exit(-1);
		}
		outfile = malloc(strlen(out_prefix));
		strncpy(outfile, out_prefix, strlen(out_prefix));
		out = fdopen(out_fd, "w+");
	}

	if ( out == NULL )
	{
		perror("Can't open temp file for saving data, bailing out");
		exit(-1);
	}

	print_runningparams(seed, simtime, Ne, Ni, outfile);

	/* temporary string used for writing onto stream */
	char outstring[128];

	/* total number of neurones; placed here to allow 
	 * command-line options to set Ne and Ni */
	const int N = Ne + Ni;

	/* time constant and total simulation time (1sec) */
	long int t;

	/* the input vector */
	double I[N];

	/* parameters of Izhikevich's model */
	double u[N], v[N], a[N], b[N], c[N], d[N];

	/* the connectivity matrix */
	double *S[N];

	/* contains the indices of neurones which fired */
	int fired[N];
	/* the total number of neurons which fired */
	int num_fired;
	
	/* allocate and init the random number generator */
	/*
	gsl_rng *rng;
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	rng = gsl_rng_alloc(T);
	gsl_rng_set(rng, seed);
	*/
	srand48((long int)seed);

	/* start measuring time */
	struct timeval t_start, t_end;
	gettimeofday(&t_start, NULL);

#pragma omp parallel 
	{
#pragma omp for nowait
		for ( i = 0 ; i < Ne ; i++ )
		{
			/* a=[0.02*ones(Ne,1);		0.02+0.08*ri]; */	
			a[i] = 0.02;
			/* b=[0.2*ones(Ne,1);		0.25-0.05*ri]; */
			b[i] = 0.2;	
			/* c=[-65+15*re.^2;		-65*ones(Ni,1)]; */
			c[i] = -65.0 + 15.0 * (genrand() * genrand());
			/* d=[8-6*re.^2;			2*ones(Ni,1)];  */
			d[i] = 8.0 - 6.0 * (genrand() * genrand());

			/* v=-65*ones(Ne+Ni,1);	% Initial values of v  */
			v[i] = -65.0;
			/* u=b.*v;					% Initial values of u  */
			u[i] = b[i] * v[i];
		}
#pragma omp for nowait
		for ( i = Ne ; i < N ; i++ )
		{
			a[i] = 0.02 + 0.08 * genrand();
			b[i] = 0.25 - 0.05 * genrand();
			c[i] = -65.0;
			d[i] = 2.0;

			/* v=-65*ones(Ne+Ni,1);	% Initial values of v  */
			v[i] = -65.0;
			/* u=b.*v;					% Initial values of u  */
			u[i] = b[i] * v[i];
		}
	}

#pragma omp parallel for shared(S) private(j)
	for ( i = 0 ; i < N ; i++ )
	{
		/* keep signedness */
		S[i] = malloc((size_t)N * sizeof &S[0]);
		if ( S[i] == NULL ) {
			perror("malloc fail!");
			exit(-1);
		}
		/* S=[0.5*rand(Ne+Ni,Ne),	-rand(Ne+Ni,Ni)]; */
		for ( j = 0 ; j < Ne ; j++ )
			S[i][j] = 0.5 * genrand();
		for ( j = Ne ; j < N ; j++ ) 
			S[i][j] = -1.0 * genrand();
	}

	for ( t = 1 ; t <= simtime ; t++ )
	{
		/* thalamic input: MUST use a normal distribution here!! */
#pragma omp parallel  
		{
#pragma omp for nowait
			for ( i = 0 ; i < Ne ; i++ )
				I[i] = genrandn() * 5.0;
#pragma omp for nowait
			for ( i = Ne ; i < N ; i++ )
				I[i] = genrandn() * 2.0;
		}

		/* find the neurones which fired (v >= 30 ) */
		j = 0;

#pragma omp barrier
//#pragma omp single
		for ( i = 0 ; i < N ; i++ )
			if ( v[i] >= 30.0 )
				fired[j++] = i;

		/* used to print to buffer */
		num_fired = j;

		/* 
		 * after-spike resetting 
		 * 
		 * in LaTeX:
		 * if v \geq 30 mV, then (group) v \larrow c \\ u \larrow u + d
		 * */
#pragma omp single
		for ( i = 0 ; i < num_fired ; i++ )
		{
			v[fired[i]] = c[fired[i]];
			u[fired[i]] += d[fired[i]];

			/* record to stream the firings neurones */
			sprintf(outstring, "%ld %d\n", t, fired[i]);
			if ( (fputs(outstring, out)) == EOF )
				perror("fputs fail");

		}

#pragma omp barrier

		/* previous FOR loop could be joined with following;
		 * however, they are kept separate to parallelise code
		 * (there is a IO write earler, no sense with parallel code)
		 */
		for ( i = 0 ; i < num_fired ; i++ )
		{
#pragma omp parallel for 
			for ( j = 0 ; j < N ; j++ )
			{
				/* 
				 * I = I + sum ( S(:, fired), 2); 
				 * a.k.a.
				 * from the matrix S
				 * extract the column of the neurones that fired
				 * and sum the rows
				 * obtain a vector of same size of I and add the two
				 * */
				I[j] += S[j][fired[i]];
			}
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
#pragma omp parallel for shared(u,v) 
		for ( i = 0 ; i < N ; i++ )
		{
			v[i] += 0.5 * ( 0.04 * (v[i]*v[i]) + 5.0 * v[i] + 140 - u[i] + I[i]);
			v[i] += 0.5 * ( 0.04 * (v[i]*v[i]) + 5.0 * v[i] + 140 - u[i] + I[i]);
			u[i] += a[i] * ( b[i] * v[i] - u[i] );
		}
	}

	gettimeofday(&t_end, NULL);
	time_t elapsed_secs = t_end.tv_sec - t_start.tv_sec;
	suseconds_t elapsed_usecs = t_end.tv_usec - t_start.tv_usec;
	printf("Elapsed time (us): %ld\n", elapsed_secs*1000000 + elapsed_usecs);

	/* don't forget to keep flushing */
	fflush(out);

	/* free stuff */
	fclose(out);
	printf("Data saved in %s\n", outfile);
	for ( i = 0 ; i < N ; i++ )
		free(S[i]);
	return 0;

}
