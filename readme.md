# Automated Product Carbon Footprint (PCF) Extraction Pipeline
**SLS Capstone Project**

📄 [Read the Full Report](./Sims_Capstone_Final_Report.pdf)

---

## The Problem

Sim's Lifecycle Services (SLS) needed to collect Product Carbon Footprint (PCF) data
from hundreds of manufacturer PDFs every reporting cycle — a process that consumed
roughly **80 labor hours** of tedious, manual transcription work every six months.

PCF reports are notoriously difficult to parse programmatically. They rely heavily on
complex visual formats like split-legend pie charts, inconsistent layouts across
manufacturers, and semi-structured graphical data that traditional tools struggle with.

---

## What We Tried

We iterated through several approaches before landing on a solution that worked:

- **Tesseract OCR** — broke down on visually complex charts
- **Azure OCR + text-only LLM** — lost semantic relationships in pie charts and graphs
- **Amazon Textract** — better layout detection, but still insufficient for graphical data
- **Hybrid OCR + NLP** — improved, but extraction reliability remained inconsistent

Each approach shared the same core flaw: decoupling the visual structure of a chart from
its meaning made it impossible to reliably extract lifecycle stage values (Manufacturing,
Transportation, Use, End-of-Life).

---

## What Worked

We pivoted to a **fully multimodal vision pipeline** built on OpenAI's API. Instead of
converting PDFs to text first, we encode pages directly as high-resolution images and
feed them into vision-capable LLMs — preserving all spatial and graphical context.

Key features of the final pipeline:
- **Three-tier escalation protocol**: starts with GPT-4o-mini for efficiency, escalates to
  GPT-4o only for complex edge cases
- **Automated validation**: algorithmic normalization, null tracking, and percentile-based
  outlier detection
- **Smart pre-processing**: deduplication and keyword-based page filtering to minimize
  unnecessary API calls

---

## Results

The pipeline reduced extraction time by over **98%**, compressing an 80-hour manual
workflow into less than one hour of unattended compute time. This frees SLS staff to
focus on analysis and client strategy rather than data transcription, with an estimated
annual savings of **~$12,000**.