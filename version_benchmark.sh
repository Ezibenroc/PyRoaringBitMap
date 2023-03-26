set -e

versions="0.3.0 0.3.2 0.3.3 0.3.4 0.3.5 0.3.6 0.3.7 0.3.8 0.4.0"

wget https://raw.githubusercontent.com/Ezibenroc/PyRoaringBitMap/benchmarking/benchmark.py

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv

python3.9 -m venv /tmp/bench
source /tmp/bench/bin/activate
pip install -q --upgrade pip
pip install -q wheel
pip install -q pandas
for i in {0..9}; do
    for ver in $(echo $versions | xargs printf "%s\n" | shuf); do
        pip install -q "pyroaring==$ver"
        # pip install --no-cache-dir https://github.com/Ezibenroc/PyRoaringBitMap/releases/download/${ver}/pyroaring-${ver}.tar.gz
        python -c 'import pyroaring; print(f"Version {pyroaring.__version__}")' || continue
        python benchmark.py --densities 0.01 0.2 0.99 --sizes 1000000 --nb_runs 1 --nb_calls 100 --add_version /tmp/bench_${ver}_${i}.csv
        pip uninstall -qy pyroaring
    done
done
deactivate
cat /tmp/bench_*.csv | head -n 1 > bench.csv
for i in /tmp/bench_*.csv; do
    cat $i | tail -n +2 >> bench.csv
done
