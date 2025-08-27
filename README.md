## Code for the paper "Characterizing Compiler Provenance from Practical and Adversarial Perspectives"

### 1. Dataset
We use the [BinKit 2.0 dataset](https://github.com/SoftSec-KAIST/BinKit) to build our experimental datasets.

### 2. General Python Scripts
* `re_arrange.py`: Re-arranging the BinKit dataset. **You should run this script ahead of all other operations since the experiments are all based on the re-arranged dataset.**
* `batch_strip.py`: Stripping the binaries within the re-arranged dataset in parallel. Note that it uses `strip -g` to retain the original function boundaries.
* `batch_binary_level_{compiler|opt}_dataset_generation.py`: Generating the experimental dataset for compiler or opt identification.

### 3. IDA Python Scripts
* `ida/batch_generation_idb.py`: Generating IDB for each binary. It requires IDA (version $\geq$ 8.2). Each IDB file is located in the same folder as the input binary.
* `ida/batch_ida_python.py`: Applying a certain IDA Python script on the given dataset in parallel.
* `ida/ida_python`: A series of IDA Python scripts for extracting features from IDB files.
* `ida/clean_intermediate_files.sh`: Cleaning all intermediate files (`*.id0`, `*.id1`, `*.id2`, `*.nam`, `*.til`, `*.$$$`) of dataset. It is for the cases that IDA exits abnormally. As a result, the expanded IDBs are not again compressed in such cases.

### 4. Ghidra Scripts
* `ghidra/batch_ghidra.py`: Run ghidra script in batches.
* `ghidra/ghidra_script`: A series of Ghidra scripts for extracting features from Ghidra projects.

### 5. Baselines
* `baselines/`: The code of the baselines we used in the paper. For some baselines, please refer to the `README.md` file in the folder for more details.

### 6. Compiler Identification
* `compiler/`: The code for compiler identification.

### 7. Optimization Identification
* `opt/`: The code for optimization level identification.
