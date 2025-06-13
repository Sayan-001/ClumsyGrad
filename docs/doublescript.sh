rm -r build/*
sphinx-build -b html source build
sphinx-build -b latex source build/latex
cd build/latex
pdflatex ClumsyGrad.tex