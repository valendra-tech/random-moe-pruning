.PHONY: all clean arxiv

LATEX      := pdflatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
SRC        := README.tex
PDF        := README.pdf
BUILD      := build
ARXIV_DIR  := $(BUILD)/arxiv
ARXIV_ZIP  := random-moe-pruning-arxiv.zip

all: $(PDF)

$(PDF): $(SRC)
	@mkdir -p $(BUILD)
	$(LATEX) $(LATEXFLAGS) -output-directory=$(BUILD) $(SRC)
	$(LATEX) $(LATEXFLAGS) -output-directory=$(BUILD) $(SRC)
	cp $(BUILD)/$(PDF) ./$(PDF)

arxiv:
	@mkdir -p $(ARXIV_DIR)
	cp README.tex $(ARXIV_DIR)/
	cp 00README.json $(ARXIV_DIR)/
	cd $(ARXIV_DIR) && zip -q -r ../../$(ARXIV_ZIP) README.tex 00README.json

clean:
	rm -rf $(BUILD) $(PDF) $(ARXIV_ZIP)
