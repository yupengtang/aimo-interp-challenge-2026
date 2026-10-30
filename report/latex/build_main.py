"""Assemble main.tex from the pandoc-converted report body and appendix."""
import re

body = open("main_body.tex", encoding="utf-8").read()
app = open("appendix.tex", encoding="utf-8").read()


def cite_once(text, needle, cite):
    assert needle in text, needle
    return text.replace(needle, needle + cite, 1)


body = cite_once(body, "runs offline on one GPU.",
                 r" The task and its labelling are described in the challenge proposal~\citep{stefanik2026aimointerp}.")
body = cite_once(body, "cases from the public MATH labels", r"~\citep{hendrycks2021math}")
body = cite_once(body, "For every problem not seen before in the run, gpt-oss-120b", r"~\citep{openai2025gptoss}")
body = cite_once(body, "720 unlabelled olympiad problems from Omni-MATH", r"~\citep{gao2024omnimath}")

# pandoc writes the appendix heading as its own \section; the template adds one
app = re.sub(r"^\\section\{.*?\}\\label\{.*?\}\s*", "", app, count=1, flags=re.S)

AI = r"""
\section*{Acknowledgements}
This research was supported in part through research cyberinfrastructure resources and services provided by the Partnership for an Advanced Computing Environment (PACE) at the Georgia Institute of Technology, Atlanta, Georgia, USA.

\section*{AI Use Statement}
I used Claude Code (Anthropic) as an assistant for ideation, coding, experiment analysis and drafting the text. I did not use generative AI tools to set the research direction or to make the final choice of which submissions to nominate and report; I made those decisions myself. I have reviewed all AI-assisted work and take full responsibility for the final content of this paper.
"""

PREAMBLE = r"""\documentclass{article}
\usepackage[sglblindworkshop, final]{neurips_2026}
\workshoptitle{AIMO Interpretability Challenge}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{longtable,array,calc}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\real}[1]{#1}
\setlength{\LTpre}{6pt}\setlength{\LTpost}{6pt}

\title{Predicting Reasoning Robustness from Hidden-State Stability}
\author{Yupeng Tang\\Georgia Institute of Technology\\\texttt{yupengtang@gatech.edu}}

\begin{document}
\maketitle
\noindent Code, data lists and submission bundles: \url{https://github.com/yupengtang/aimo-interp-challenge-2026}. Codabench participant \texttt{ytang}.\par\medskip
"""

tex = (PREAMBLE + body + AI
       + "\n\\bibliographystyle{plainnat}\n\\bibliography{references}\n\n\\appendix\n\\section{Reproducing the methods}\n"
       + app + "\n\\end{document}\n")
with open("main.tex", "w", encoding="utf-8", newline="\n") as fh:
    fh.write(tex)
print("ok")
