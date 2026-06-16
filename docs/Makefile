.PHONY: all clean

LATEX      := pdflatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
SRC        := README.tex
PDF        := README.pdf
BUILD      := build

all: $(PDF)

$(PDF): $(SRC)
	@mkdir -p $(BUILD)
	$(LATEX) $(LATEXFLAGS) -output-directory=$(BUILD) $(SRC)
	$(LATEX) $(LATEXFLAGS) -output-directory=$(BUILD) $(SRC)
	cp $(BUILD)/$(PDF) ./$(PDF)

clean:
	rm -rf $(BUILD) $(PDF)
