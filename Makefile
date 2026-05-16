.PHONY: pdf figures clean help

PAPER_MD     := paper/PAPER.md
PAPER_PDF    := paper/PAPER.pdf
FIGURES      := paper/figures/fig_hit_rate_vs_k.png \
                paper/figures/fig_cost_per_user_vs_k.png \
                paper/figures/fig_cost_ratio_vs_k.png \
                paper/figures/fig_cost_breakdown.png \
                paper/figures/fig_theoretical_scaling.png

help:
	@echo "Targets:"
	@echo "  make pdf      - render paper/PAPER.md to paper/PAPER.pdf via pandoc + xelatex"
	@echo "  make figures  - regenerate figures from runs/multitenant_*.jsonl"
	@echo "  make clean    - remove PAPER.pdf"

pdf: $(PAPER_PDF)

PAPER_HEADER := paper/header.tex

$(PAPER_PDF): $(PAPER_MD) $(PAPER_HEADER) $(FIGURES)
	pandoc $(PAPER_MD) \
	  --resource-path=paper \
	  --include-in-header $(PAPER_HEADER) \
	  -o $(PAPER_PDF) \
	  --pdf-engine=xelatex \
	  -V geometry:margin=1in \
	  -V fontsize=11pt \
	  -V linkcolor=blue \
	  -V urlcolor=blue \
	  -V documentclass=scrartcl \
	  -V classoption=DIV=12 \
	  -V mainfont=Cambria \
	  -V sansfont=Cambria \
	  -V monofont="Cascadia Mono" \
	  --toc --toc-depth=2 \
	  --standalone

figures:
	python paper/figures/plot.py

clean:
	rm -f $(PAPER_PDF)
