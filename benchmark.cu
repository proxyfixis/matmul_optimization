%%writefile benchmark.cu
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

struct Result {
    string name;
    double time_ms;
    double gflops;
    double cublas_gflops;
    double efficiency;
};

Result run_and_parse(const string &cmd) {
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        cerr << "Failed to run " << cmd << endl;
        exit(1);
    }

    char buffer[512];
    Result res;
    res.name = cmd;

    while (fgets(buffer, sizeof(buffer), pipe)) {
        if (strstr(buffer, "RESULT:")) {
            // Parse the RESULT line
            char name[64];
            sscanf(buffer,
                   "RESULT: KernelName=%63[^,], AvgTime=%lf, GFLOPS=%lf, CuBLAS=%lf, Efficiency=%lf",
                   name, &res.time_ms, &res.gflops, &res.cublas_gflops, &res.efficiency);
            res.name = name;
        }
    }

    pclose(pipe);
    return res;
}

int main() {
    // List all binaries
    vector<string> cmds = {
        "./naive",
        "./tiled4096",
        "./tiledregister4096",
        "./doublebuffered"
    };

    vector<Result> results;
    for (auto &cmd : cmds) {
        cout << "Running " << cmd << " ..." << endl;
        results.push_back(run_and_parse(cmd));
    }

    // --- Compute derived values ---
    double naive_time = results[0].time_ms;
    double cublas_gflops = results.back().cublas_gflops; // from last run

    // --- Print summary table ---
    cout << "\n================ Performance Summary ================\n";
    cout << "| Implementation         | Time (ms) |  GFLOPS  | % of cuBLAS | Speedup vs Naive |\n";
    cout << "|------------------------|-----------|-----------|--------------|------------------|\n";
    for (auto &r : results) {
        double speedup = naive_time / r.time_ms;
        double percent = (r.gflops / cublas_gflops) * 100.0;
        printf("| %-22s | %9.3f | %9.2f | %9.2f%% | %6.2fx |\n",
               r.name.c_str(), r.time_ms, r.gflops, percent, speedup);
    }
    cout << "========================================================\n";
    return 0;
}
