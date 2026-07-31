# pip install git+https://github.com/Starmys/flash-attention.git@weighted
git clone -b weighted https://github.com/Starmys/flash-attention.git
cd flash-attention && pip install . --no-build-isolation && cd ..

cd library/
git clone https://github.com/NVIDIA/cutlass.git
cd retroinfer && pip install . --no-build-isolation && cd ..