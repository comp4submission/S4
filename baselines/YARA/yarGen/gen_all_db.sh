python yarGen.py -c -g ~/train_for_yara_compiler/clang-10.0 -i compiler-clang-10.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-11.0 -i compiler-clang-11.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-12.0 -i compiler-clang-12.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-13.0 -i compiler-clang-13.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-4.0  -i compiler-clang-4.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-5.0  -i compiler-clang-5.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-6.0  -i compiler-clang-6.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-7.0  -i compiler-clang-7.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-8.0  -i compiler-clang-8.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/clang-9.0  -i compiler-clang-9.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/gcc-10.3.0 -i compiler-gcc-10.3.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/gcc-11.2.0 -i compiler-gcc-11.2.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/gcc-4.9.4  -i compiler-gcc-4.9.4 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/gcc-5.5.0  -i compiler-gcc-5.5.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/gcc-6.5.0  -i compiler-gcc-6.5.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/gcc-7.3.0  -i compiler-gcc-7.3.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/gcc-8.2.0  -i compiler-gcc-8.2.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_compiler/gcc-9.4.0  -i compiler-gcc-9.4.0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_opt/O0 -i opt-O0 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_opt/O1 -i opt-O1 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_opt/O2 -i opt-O2 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_opt/O3 -i opt-O3 --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_opt/Ofast -i opt-Ofast --opcodes -fs 500
python yarGen.py -c -g ~/train_for_yara_opt/Os -i opt-Os --opcodes -fs 500
mv dbs all_dbs
mkdir dbs
