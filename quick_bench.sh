classes="pybitmap cybitmap set"

# Sets are equal to range(b, N, U), with b either 0 or 1
N=100000000
U=8
common_stmt="from pyroaring import BitMap as pybitmap; from roaringbitmap import RoaringBitmap as cybitmap; import random; import pickle; my_set="
init_stmt=";range_val=range(0,$N,$U); list_val=list(range_val)"
for stmt in "my_set()" "my_set(range_val)" "my_set(list_val)"; do
    echo "$stmt"
    for cls in $classes; do
        t=`python3 -m timeit -u msec -s "$common_stmt$cls$init_stmt" "$stmt" | cut -d" " -f6`
        echo "    $cls $t"
    done
done
init_stmt=";val=random.randint(0, $N/2); s1=my_set(range(0,$N,$U));s2=my_set(range(1,$N,$U));s3=my_set(range(0,$N,$U));size=len(s1)"
for stmt in "s1.add(val)" "val in s1" "x=list(s1)" "s1==s3" "s1 <= s3" "x=s1|s2" "x=s1&s2" "x=s1^s2" "pickle.loads(pickle.dumps(s1))"; do
    echo "$stmt"
    for cls in $classes; do
        t=`python3 -m timeit -u msec -s "$common_stmt$cls$init_stmt" "$stmt" | cut -d" " -f6`
        echo "    $cls $t"
    done
done
for stmt in "s1[int(size/2)]" "s1[int(size/4):3*int(size/4):2]"; do
    echo "$stmt"
    for cls in "pybitmap" "cybitmap"; do
        t=`python3 -m timeit -u msec -s "$common_stmt$cls$init_stmt" "$stmt" | cut -d" " -f6`
        echo "    $cls $t"
    done
done
