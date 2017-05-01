bash CRoaring/amalgamation.sh
echo "//   Copyright `date +%Y` The CRoaring authors
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
// Github repository: https://github.com/RoaringBitmap/CRoaring/
// Official website : http://roaringbitmap.org/
" > /tmp/roaring.c
cp /tmp/roaring.c /tmp/roaring.h
cat roaring.c >> /tmp/roaring.c
cat roaring.h >> /tmp/roaring.h
rm roaring.c roaring.h
sed -E "s|#include \"roaring\.h\"|#include \"roaring\.hh\"|g" /tmp/roaring.c > roaring.cpp
cp -f /tmp/roaring.h roaring.hh
