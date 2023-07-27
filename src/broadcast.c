#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

#define NOT_READ -1
#define MAX_VAL 1000

// Procedimento auxiliar para inicializar o gerador de 
// numeros pseudo-aleatorios (PRNG).
// Utiliza-se o Unix time em microssegundos (us) como seed.
void initialize_prng() {
    struct timeval tv;
    long total_us;

    gettimeofday(&tv, NULL);
    total_us = tv.tv_sec * 1000000 + tv.tv_usec;

    srand(total_us);
}

void print_usage_message() {
    printf("Uso: mpirun -np <num_procs> ");
    printf("./broadcast --array_size <array_size> [--root <root>] [--custom]\n");
    printf("   num_procs: numero de processos MPI\n");
    printf("   array_size: tamanho do array a ser distribuido entre processos\n");
    printf("   root: rank do processo root [default = 0]\n");
    printf("   --custom: executa custom_bcast no lugar de MPI_Bcast\n");
}

void parse_arguments(
    int argc,
    char *argv[],
    int *array_size_ptr,
    bool *is_custom_bcast_ptr,
    int *root_ptr) {

    int ret;

    if (argc < 3 || argc > 6) {
        print_usage_message();
        exit(EXIT_FAILURE);
    }

    *array_size_ptr = NOT_READ;
    *is_custom_bcast_ptr = false;
    *root_ptr = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--custom") == 0) {
            *is_custom_bcast_ptr = true;
        }

        if (i < argc-1) {
            if (strcmp(argv[i], "--array_size") == 0) {
                ret = sscanf(argv[i+1], "%d", array_size_ptr);
                if (ret == 0 || ret == EOF) {
                    fprintf(stderr, "Error reading array_size\n");
                    exit(EXIT_FAILURE);
                }
            }

            if (strcmp(argv[i], "--root") == 0) {
                ret = sscanf(argv[i+1], "%d", root_ptr);
                if (ret == 0 || ret == EOF) {
                    fprintf(stdout, "Error reading root, setting to default value 0\n");
                    *root_ptr = 0;
                }
            }
        }
    }

    if (*array_size_ptr == NOT_READ) {
        print_usage_message();
        exit(EXIT_FAILURE);
    }
}

int custom_bcast(
    void *buffer,
    int count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm comm ) {

    // Variáveis para armazenar o número de processos e o rank do processo atual
    int size, rank, i;
    MPI_Status status;

    // Obtém o número de processos
    MPI_Comm_size(comm, &size);
    // Obtém o rank do processo atual
    MPI_Comm_rank(comm, &rank);

    // Verifica se o processo atual é o processo root
    if (rank == root) {
        // Se sim, então ele envia o buffer para todos os outros processos
        for(i = 0; i < size; i++) {
            // Exclui o processo root da operação de envio
            if(i != rank) {
                // Envia o buffer para o processo 'i'
                int err = MPI_Send(buffer, count, datatype, i, 0, comm);
                // Verifica se houve erro no envio
                if (err != MPI_SUCCESS) {
                    // Se sim, imprime uma mensagem de erro e retorna o código de erro
                    fprintf(stderr, "Error sending data to process %d\n", i);
                    return err;
                }
            }
        }
    } else {
        // Se o processo atual não é o root, ele deve receber o buffer do processo root
        int err = MPI_Recv(buffer, count, datatype, root, 0, comm, &status);
        // Verifica se houve erro no recebimento
        if (err != MPI_SUCCESS) {
            // Se sim, imprime uma mensagem de erro e retorna o código de erro
            fprintf(stderr, "Error receiving data from root process %d\n", root);
            return err;
        }
    }

    // Se todas as operações de envio e recebimento foram bem-sucedidas, retorna MPI_SUCCESS
    return MPI_SUCCESS;

}

int main(int argc, char *argv[]) {
    int rank;

    // Inicializacao do ambiente de execucao MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Leitura dos argumentos de linha de comando
    int array_size, root; 
    bool is_custom_bcast;
    parse_arguments(argc, argv, &array_size, &is_custom_bcast, &root);

    // Preenche buffer no processo root
    // com inteiros entre 0 e MAX_VAL.
    // Usando o operador % pois nao estamos preocupados
    // se os valores possuem distribuicao uniforme.
    // O limite MAX_VAL eh arbitrario, apenas para facilitar 
    // a verificacao do resultado.
    int *buf = (int *) malloc(array_size * sizeof(int));
    if (!buf) {
        fprintf(stderr, "Error allocating buffer of size %d\n", array_size);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    if (rank == root) {
        initialize_prng();
        for (int i = 0; i < array_size; i++)
            buf[i] = rand() % MAX_VAL;
    }

    // Chamada a MPI_Bcast ou custom_bcast e
    // medicao do tempo transcorrido
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (is_custom_bcast)
        custom_bcast(buf, array_size, MPI_INT, root, MPI_COMM_WORLD);
    else
        MPI_Bcast(buf, array_size, MPI_INT, root, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Imprime dados em cada processo no modo debug
    #ifdef DEBUG
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    for (int r = 0; r < num_procs; r++) {
        if (r == rank) {
            if (r == root)
                printf("Dados enviados do processo %d (root)\n", r);
            else
                printf("Dados recebidos no processo %d\n", r);

            for (int i = 0; i < array_size; i++)
                printf("%d ", buf[i]);

            printf("\n\n");
        }

        // Para garantir a ordem de impressao
        MPI_Barrier(MPI_COMM_WORLD);
    }
    #endif

    // Processo root imprime o tempo transcorrido
    // na operacao broadcast, em segundos
    if (rank == root)
        printf("%lf\n", t2 - t1);

    free(buf);
    MPI_Finalize();
    return 0;
}
