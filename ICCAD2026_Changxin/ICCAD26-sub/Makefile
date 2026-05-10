ENGINE := pdflatex
BIB := bibtex

PROJECT := ISEDA25-MLAE

all:
	$(ENGINE) $(PROJECT).tex
	$(BIB)    $(PROJECT)
	$(BIB)    $(PROJECT)
	$(ENGINE) $(PROJECT).tex
	$(ENGINE) $(PROJECT).tex

fast:
	$(ENGINE) $(PROJECT).tex


clean:
	rm *.log *.aux *.blg *.bbl *.gz *.dvi *.thm *.dat *.out
