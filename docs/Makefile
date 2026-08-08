# Makefile for building ClumsyGrad documentation

SPHINXOPTS  ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = source
BUILDDIR    = build

.PHONY: help html pdf clean

help:
	@echo "Available targets:"
	@echo "  html   Build HTML documentation into $(BUILDDIR)/html"
	@echo "  pdf    Build PDF documentation into $(BUILDDIR)/latex"
	@echo "  clean  Remove the build directory"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

pdf:
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)
	cd "$(BUILDDIR)/latex" && pdflatex -interaction=nonstopmode ClumsyGrad.tex && pdflatex -interaction=nonstopmode ClumsyGrad.tex

clean:
	rm -rf "$(BUILDDIR)"
