.PHONY: pdf diagrams figures clean help

PAPER_MD      := paper/PAPER.md
PAPER_PDF     := paper/PAPER.pdf

DIAGRAM_SRCS  := $(wildcard paper/diagrams/*.mmd)
DIAGRAM_PNGS  := $(DIAGRAM_SRCS:.mmd=.png)
MERMAID_CFG   := paper/diagrams/mermaid-config.json

FIGURE_PNGS   := paper/figures/fig_hit_rate_vs_k.png \
                 paper/figures/fig_cost_per_user_vs_k.png \
                 paper/figures/fig_cost_ratio_vs_k.png \
                 paper/figures/fig_cost_breakdown.png \
                 paper/figures/fig_theoretical_scaling.png

help:
	@echo "Targets:"
	@echo "  make pdf       - render paper/PAPER.md to paper/PAPER.pdf"
	@echo "  make diagrams  - render Mermaid architecture diagrams (.mmd -> .png)"
	@echo "  make figures   - regenerate benchmark figures from runs/*.jsonl"
	@echo "  make clean     - remove paper/PAPER.pdf"

pdf: $(PAPER_PDF)

$(PAPER_PDF): $(PAPER_MD) $(DIAGRAM_PNGS) $(FIGURE_PNGS)
	pandoc $(PAPER_MD) \
	  --resource-path=paper \
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

diagrams: $(DIAGRAM_PNGS)

paper/diagrams/%.png: paper/diagrams/%.mmd $(MERMAID_CFG)
	mmdc -i $< -o $@ -c $(MERMAID_CFG) -s 2 -b white

figures:
	python paper/figures/plot.py

clean:
	rm -f $(PAPER_PDF)
