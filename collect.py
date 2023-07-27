import csv
import subprocess
from dataclasses import dataclass, fields


@dataclass
class Result:
    array_exp: int
    threads: int
    custom: bool
    root: int
    time: float
    stddev: float

    # def __post_init__(self):
    #     self.array_size = 2**self.array_exp

    @staticmethod
    def __parse(key, value):
        if key in ["array_exp", "threads", "root"]:
            return int(value)
        elif key in ["time", "stddev"]:
            return float(value)
        elif key == "custom":
            return value == "True"
        else:
            raise ValueError(f"Unknown key {key}")

    @staticmethod
    def from_dict(d):
        return Result(**{k: Result.__parse(k, v) for k, v in d.items()})


class MonteCarlo:
    def __init__(self, max_exp, max_threads, runs):
        import math

        self.max_exp = max_exp
        self.max_thread_exp = int(math.log2(max_threads))
        self.runs = runs
        filename_postfix = f"{max_exp}me_{max_threads}mt_{runs}runs"
        self.results_path = f"data/data_{filename_postfix}.csv"
        self.plot_path = f"plots/plot_{filename_postfix}.png"

    def run(self):
        print("Running Monte Carlo")

        subprocess.call(["make"], cwd="src")

        results = []
        for custom in [False, True]:
            for thread_exp in range(self.max_thread_exp + 1):
                threads = 2**thread_exp
                for array_exp in range(10, self.max_exp + 1):
                    print(f"threads: {threads}, array_exp: {array_exp}, custom: {custom}")
                    result = self.__simulate(array_exp, threads, custom)
                    results.append(result)
                    print(result.time)

        with open(self.results_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Result)])
            writer.writeheader()

            rows = [r.__dict__ for r in results]
            writer.writerows(rows)

        return results

    def __run(self, array_size, threads, custom, root):
        from multiprocessing import cpu_count as thread_count
        from psutil import cpu_count

        if root >= threads:
            raise ValueError(f"{root=} must be less than {threads=}")

        additional_flags = []
        if threads > thread_count():
            additional_flags.append("--oversubscribe")
        elif threads > cpu_count(logical=False):
            additional_flags.append("--use-hwthread-cpus")

        # mpirun -np $threads ./broadcast --array_size $k --root $root
        process = subprocess.run(["mpirun",
                                  "-np", str(threads),
                                  *additional_flags,
                                  "./broadcast",
                                  "--array_size", str(array_size),
                                  "--root", str(root),
                                  "--custom" if custom else ""],
                                 text=True, capture_output=True, cwd="src")
        return process.stdout.strip()

    def __simulate(self, array_exp, threads, custom, root=0):
        times = []
        for _ in range(self.runs):
            time = self.__run(2**array_exp, threads, custom, root)
            time = float(time)
            assert time >= 0, f"{time=} is not positive"
            time /= 1000
            times.append(time)
        time = sum(times) / self.runs
        stddev = sum((t - time)**2 for t in times) / self.runs
        return Result(array_exp, threads, custom, root, time, stddev)

    def load(self):
        from os.path import exists

        if not exists(self.results_path):
            raise FileNotFoundError("%s not found, please run simulation first", self.results_path)

        print("Loading from runs.csv")
        with open(self.results_path) as f:
            reader = csv.DictReader(f)
            runs = list(reader)

        runs = list(map(Result.from_dict, runs))

        return runs

    def plot(self, runs):
        from matplotlib import pyplot as plt

        print("Plotting results")

        ncols = 3
        if self.max_thread_exp <= 4:
            ncols = 2
            nrows = 2
        elif self.max_thread_exp <= 6:
            nrows = 2
        else:
            nrows = 3

        _, axes = plt.subplots(nrows, ncols, figsize=(10, 10))

        for i, thread_exp in enumerate(range(self.max_thread_exp + 1)):
            threads = 2**thread_exp

            run = [r for r in runs if r.threads == threads]
            row = i // ncols
            col = i % ncols

            default = [r for r in run if not r.custom]
            custom = [r for r in run if r.custom]

            axes[row, col].errorbar([r.array_exp for r in default],
                                    [r.time for r in default],
                                    yerr=[r.stddev for r in default],
                                    color='blue', marker='^',
                                    linestyle='none', label='MPI')

            axes[row, col].errorbar([r.array_exp for r in custom],
                                    [r.time for r in custom],
                                    yerr=[r.stddev for r in custom],
                                    color='red', marker='o',
                                    linestyle='none', label='custom')

            axes[row, col].set_title(f"{threads=}")
            axes[row, col].set_xlabel("log2(array size)")
            axes[row, col].set_ylabel("time (ms)")

            axes[row, col].ticklabel_format(style='sci', scilimits=(-3, 3))
            axes[row, col].legend()

        plt.savefig(self.plot_path, dpi=300)


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Monte Carlo simulation")
    parser.add_argument("-mt", "--max-threads", type=int, default=32,
                        help="Maximum number of threads to use")
    parser.add_argument("-me", "--max-exp", type=int, default=17,
                        help="Maximum exponent of array size to use")
    # parser.add_argument("-r", "--root", type=int, default=0,
    #                     help="Root node to broadcast from")
    parser.add_argument("-r", "--runs", type=int, default=100,
                        help="Number of runs to average over")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot results")

    return parser.parse_args()


def main():
    from os import mkdir
    from os.path import exists

    if not exists("src"):
        raise FileNotFoundError("src/ not found, please run from project root directory")

    if not exists("data"):
        mkdir("data")

    args = parse_args()

    sim = MonteCarlo(args.max_exp, args.max_threads, args.runs)
    if input("Run Monte Carlo? (y/n) ") == "y":
        runs = sim.run()
    else:
        runs = sim.load()

    if args.plot:
        if not exists("plots"):
            mkdir("plots")
        sim.plot(runs)


if __name__ == "__main__":
    main()
