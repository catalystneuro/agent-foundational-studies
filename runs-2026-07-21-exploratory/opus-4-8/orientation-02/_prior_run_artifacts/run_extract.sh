set -e
declare -a A=(
"02291b99-e583-498b-9929-b68bba2c50e2 719161530"
"b4aeeb19-cdc6-4895-ab7b-bc8a688cf6f5 732592105"
"96c200cf-29c2-457a-b2f3-99f11de5b039 742951821"
"286c7b06-3cde-4261-9090-e6fbe6c81945 750332458"
"be9f8fd8-8f16-4a66-acc6-9e04697650f3 751348571"
"4513d0c9-1e2b-4c8a-aa22-ae81822537c9 760693773"
"d6f3e82b-aca9-43fb-9810-048fc2124d50 762120172"
)
for x in "${A[@]}"; do
  set -- $x
  if [ ! -f "cache_$2.pkl" ]; then
    echo "=== $2"
    python 03_extract.py "$1" "cache_$2.pkl" 2>&1 | grep -v "it/s\]" | tail -3
  fi
done
echo ALLDONE
