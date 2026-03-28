

**RESEARCH PROPOSAL**

**Fin-JEPA**

Cross-Modal Coherence as a Financial Risk Signal

A Joint Embedding Predictive Architecture with

Data Efficiency Bounds for Low-Resource Markets

**Elroy Galbraith**

Chief Data Officer, Aeon Technology Solutions

March 2026

Industry Partner: Aeon Technology Solutions

Primary Data: SEC EDGAR (public)

Validation Data: Jamaica Stock Exchange (proposed Study 3\)

*Version 1.0*

# **1\. Abstract**

This proposal outlines a research programme to develop Fin-JEPA, a Joint Embedding Predictive Architecture for financial risk assessment that introduces cross-modal coherence—the degree to which a company’s narrative disclosures and quantitative financials agree in learned embedding space—as a novel risk signal. The architecture uses dual context-target encoder pairs for each modality: a document encoder initialised from a finance-tuned language model and an FT-Transformer financial encoder, with both encoder pairs evolving during training through the JEPA self-supervised objective. A lightweight predictor learns cross-modal relationships in the resulting latent space.

The research is structured as a sequence of four studies, each validating the assumptions the next one rests on. Study 0 establishes the FT-Transformer financial encoder as a viable representation learner for XBRL data, de-risking the tabular component before coupling it with the document encoder. Study 1 introduces the full Fin-JEPA architecture and cross-modal coherence signal, evaluated on the complete SEC EDGAR corpus against rigorous baselines—including a simple sentiment-ratio divergence metric and contemporaneous market-based signals—with dedicated tests for the complexity confound, market signal redundancy, and financial encoder contribution. Study 2 conducts systematic ablation—reducing company count, history depth, and trading density—to produce data efficiency bounds and a scaling surface that maps the viability boundary for markets of varying sizes. Study 3 (contingent on encouraging ablation results) validates the approach on the Jamaica Stock Exchange or another frontier market, preceded by a formal data quality audit of the extraction pipeline.

The coherence signal produces a per-company continuous risk score that can be thresholded for binary classification of specific adverse events—earnings restatements, audit qualifications, enforcement actions, bankruptcy—and decomposes directionally into narrative inflation (management overstating) and hidden risk (financial patterns not disclosed) components. This incremental structure ensures that each study produces a publishable, self-contained contribution while establishing the empirical foundation for the next. A negative result at any stage is informative rather than catastrophic: the community learns where and why the approach breaks down, and resources are not committed to downstream studies whose assumptions have been invalidated.

# **2\. Problem Statement**

Financial risk assessment combines quantitative signals (ratios, trends, cash flows) with qualitative judgment (management credibility, strategic coherence, disclosure quality). Existing machine learning approaches treat these modalities separately or concatenate them naïvely, missing the interaction effect. A company whose management paints a rosy picture while the balance sheet deteriorates presents a signal that emerges only at the intersection of modalities.

Joint Embedding Predictive Architectures (JEPA), proposed by LeCun (2022), offer a principled framework for learning cross-modal relationships in latent space. By training a predictor to map from one modality’s embedding to another’s, JEPA captures what statistical consistency looks like across data surfaces—and, critically, produces a natural measure of inconsistency when the predictor fails. In the financial context, this prediction error is the cross-modal coherence signal: the degree to which the numbers and the narrative agree.

An important caveat: this cross-modal prediction error captures statistical co-occurrence patterns in embedding space, which is a different operation from the causal reasoning that human analysts employ when they judge whether a company’s story “adds up.” Whether the learned signal corresponds to the kind of inconsistency that experienced analysts detect is an empirical question this research explicitly investigates through interpretability analysis, not a premise it assumes.

## **2.1 Why JEPA Over Contrastive Learning**

A natural alternative would be a CLIP-style contrastive approach that learns to match financial and narrative embeddings for the same company-year while pushing apart unpaired observations. However, contrastive learning produces a symmetric similarity score, whereas JEPA’s predictive objective produces directional prediction errors: the financial encoder’s failure to predict the document embedding (suggesting narrative inflation) is a distinct signal from the document encoder’s failure to predict the financial embedding (suggesting hidden financial risk). This directionality is central to the proposed contribution—the coherence signal decomposes into interpretable components precisely because the objective is predictive rather than contrastive. Additionally, contrastive methods require careful negative sampling, which is non-trivial in financial data where companies in the same sector and year share substantial structural similarity; JEPA avoids this requirement entirely.

The JEPA paradigm has been demonstrated in computer vision (I-JEPA, V-JEPA), video understanding (V-JEPA 2), and vision-language tasks (VL-JEPA), but has not been applied to financial data. The financial domain presents a distinct challenge: the modalities are heterogeneous (tabular financials vs. natural language), the signal-to-noise ratio is low, the outcome of interest (financial distress) is rare, and the available data varies dramatically in scale across markets.

This raises two questions the literature has not addressed. First, can JEPA’s cross-modal prediction objective learn meaningful representations from financial data, and does the resulting coherence signal predict real-world risk beyond what simpler approaches capture? Second, how does this signal scale with data availability—specifically, what are the minimum corpus size and history depth at which the approach remains viable? The second question is critical for determining whether the methodology transfers to the low-resource markets where automated risk intelligence is arguably most needed.

# **3\. Research Questions**

## **3.1 Primary Research Question**

Can a Joint Embedding Predictive Architecture trained on multi-modal financial data learn cross-modal representations whose prediction error (the coherence signal) is predictive of subsequent financial distress, and how does this predictive power scale with corpus size?

## **3.2 Secondary Research Questions**

* **Financial Encoder Viability:** Can an FT-Transformer trained on XBRL financial statement data learn representations that are competitive with gradient-boosted trees and traditional ratios for distress prediction? This foundational question must be answered before coupling the financial encoder with the document encoder in the full JEPA architecture.

* **Cross-Modal Coherence as Risk Signal:** Does high prediction error between the financial encoder and the document encoder correlate with subsequent adverse events? Critically, is this signal additive to both traditional financial ratios and a simple divergence baseline (e.g., sentiment-score vs. financial-health composite disagreement)?

* **Market Signal Redundancy:** Does the coherence signal provide lead time over market-based signals, or does it merely confirm what contemporaneous price data (momentum, volatility, short interest) already reflects? Regulatory filings are backward-looking and published with a lag; the signal must demonstrate incremental value after controlling for information already available through trading data.

* **Complexity Confound:** Does the coherence signal capture genuine narrative-financial inconsistency, or is it primarily a proxy for business complexity? This is tested by controlling for complexity measures and by examining whether within-firm changes in coherence predict distress more strongly than cross-sectional levels. Complexity and coherence may be partially overlapping constructs—the analysis decomposes their relative contributions rather than discarding either.

* **Financial Encoder Contribution:** Does the financial modality make a genuine contribution to the coherence signal, or does the predictor learn to generate plausible financial embeddings from text features alone? This is tested by replacing the financial encoder input with noise and measuring performance degradation.

* **Data Efficiency Bounds:** What is the scaling behaviour of the coherence signal’s predictive power as a function of corpus size and history depth? At what thresholds does performance degrade below the best non-JEPA baseline? These thresholds define the minimum viable corpus for frontier market deployment.

* **Distributional Shift Under Ablation:** Do ablation results hold when the downsampled corpus is drawn with frontier-market-like sector weights (e.g., heavy financials, minimal technology), rather than preserving the US large-cap sector distribution?

* **Modality Contribution:** What is the relative contribution of each prediction direction (financial-to-document vs. document-to-financial) to the coherence signal? Does the signal emerge primarily from detecting narrative inflation or from detecting hidden risk?

* **Domain Adaptation Value:** How do representations from a general-purpose language model compare to a finance-tuned encoder (e.g., FinBERT), and what is the marginal value of further regional adaptation?

# **4\. Aims and Objectives**

## **4.1 Aim**

To develop, validate, and open-source a Joint Embedding Predictive Architecture for financial risk assessment that establishes cross-modal coherence as a novel risk signal and produces empirically grounded data efficiency bounds determining the approach’s viability for capital markets of varying sizes.

## **4.2 Objectives**

* Validate the FT-Transformer as a viable financial encoder on SEC XBRL data, establishing baseline tabular representation quality before cross-modal coupling (Study 0).

* Construct a large-scale research dataset from SEC EDGAR, comprising 10-K/10-Q filings with XBRL-extracted financials, pre-parsed MDA sections, and corresponding market data.

* Design and implement the Fin-JEPA architecture with a finance-tuned document encoder, FT-Transformer financial encoder with optional self-supervised pretraining, and a cross-modal latent-space predictor.

* Evaluate the cross-modal coherence signal against established baselines including a simple sentiment-ratio divergence metric and contemporaneous market-based signals, with outcomes disaggregated by distress type (Study 1).

* Verify financial encoder contribution through noise-replacement ablation and test for market signal redundancy through controls for contemporaneous trading data (Study 1).

* Conduct systematic ablation studies with both sector-preserving and frontier-market-weighted sampling to establish data efficiency bounds (Study 2).

* Contingent on encouraging ablation results, validate the approach on JSE or another frontier market, preceded by a formal data quality audit of the extraction pipeline (Study 3).

* Publish findings at each stage, release model weights, training code, data extraction pipelines, universe definitions, and train/test split specifications as open-source.

# **5\. Research Programme Structure**

The programme is decomposed into four studies, each validating the assumptions the next one depends on. This incremental design reduces risk, produces publishable outputs at each stage, and allows course-correction based on findings.

## **5.0 Study 0: Financial Encoder Validation**

### **5.0.1 Motivation**

The full Fin-JEPA architecture couples a pretrained document encoder (FinBERT) with a randomly-initialised FT-Transformer financial encoder. This creates an asymmetry: the document encoder arrives with a rich embedding space from millions of documents, while the financial encoder must learn everything from the training corpus. If the FT-Transformer cannot learn useful financial representations from XBRL data in isolation, the cross-modal architecture inherits that weakness—and any failure in Study 1 would be ambiguous between “the cross-modal objective doesn’t work” and “the financial encoder never produced useful inputs.”

Study 0 eliminates this ambiguity. It also investigates whether a self-supervised pretraining phase for the FT-Transformer—training it to predict masked or corrupted financial features before JEPA training begins—produces a stronger initialisation that mitigates the asymmetry with the pretrained document encoder.

### **5.0.2 Design**

The FT-Transformer is trained on XBRL financial statement data from the full SEC corpus for supervised distress prediction. Performance is benchmarked against XGBoost on the same features (the standard tabular baseline), logistic regression on traditional financial ratios (Altman Z-score components, current ratio, leverage, etc.), and a gradient-boosted tree ensemble on raw XBRL features.

A secondary experiment applies self-supervised pretraining to the FT-Transformer: randomly masking subsets of financial features and training the model to reconstruct them. The pretrained encoder is then fine-tuned for distress prediction and compared against the randomly-initialised version. If self-supervised pretraining improves performance, it becomes the default initialisation strategy for the financial encoder in Studies 1–2.

### **5.0.3 Success Criteria**

The FT-Transformer must match or exceed XGBoost on distress prediction to justify its use in the cross-modal architecture. If it substantially underperforms, the architecture design must be revisited before proceeding to Study 1\.

### **5.0.4 Output**

A short workshop paper or technical report establishing FT-Transformer viability on financial tabular data, with or without self-supervised pretraining.

## **5.1 Study 1: Cross-Modal Coherence Signal**

### **5.1.1 Motivation**

This is the core contribution: demonstrating that JEPA prediction error between financial and narrative modalities constitutes a meaningful risk signal. Study 1 answers one clean question: does the coherence signal predict distress, and does it add value beyond simpler approaches?

### **5.1.2 Architecture**

The Fin-JEPA architecture uses dual context-target encoder pairs for each modality, following the JEPA paradigm. Each modality has a context encoder (trained via backpropagation) and a target encoder (updated as an exponential moving average of the context encoder). A lightweight predictor maps from one modality’s context embedding to the other’s target embedding bidirectionally.

**Document Encoder.** Initialised from FinBERT or a finance-adapted sentence transformer. Both context and EMA target copies begin from the same pretrained checkpoint. During training, the context encoder’s weights are updated via backpropagation from the prediction loss, while the target encoder tracks via EMA. The pretrained initialisation provides financial language understanding; training reshapes this toward cross-modal predictability.

**Financial Encoder.** FT-Transformer architecture with per-feature tokenisation, initialised either randomly or from the self-supervised pretraining developed in Study 0, depending on those results. Both context and EMA target copies are maintained.

**Predictor.** A lightweight network (small transformer or MLP) that maps between modality embeddings bidirectionally. The bidirectional loss flows gradients back through the predictor and into the context encoders but not into the target encoders.

**Collapse Prevention.** The EMA target encoder is the primary mechanism. Additional regularisation via VICReg or similar may be applied as determined through experimentation. Importantly, the regularisation strategy must be calibrated to avoid smoothing over the very incoherence the signal is designed to detect—over-regularisation risks suppressing outlier embeddings that correspond to genuine narrative-financial divergence.

### **5.1.3 Evaluation Tiers**

The evaluation proceeds in three tiers, each measuring a distinct aspect of the learned representations. This structure ensures that evaluation is interpretable even if the coherence signal underperforms on downstream prediction: embedding quality can be assessed independently of predictive power.

**Tier 1: Intrinsic Embedding Quality.** The quality of the learned embedding space is assessed directly, without reference to outcome labels. Metrics include sector clustering purity (whether companies in the same industry cluster together in embedding space), temporal coherence (whether the same company’s embeddings across adjacent years are proximal), and distress separation (whether companies that subsequently experienced adverse events occupy a distinct region). Visualisation via UMAP or t-SNE complements quantitative metrics (silhouette scores, nearest-neighbour purity). This tier answers: did the model learn structured, meaningful representations?

**Tier 2: Coherence Signal as Risk Indicator.** The cross-modal prediction error for each company-year in a held-out test set is computed and evaluated as a continuous risk score. The analysis uses logistic regression (for binary adverse events) and Cox proportional hazards regression (for time-to-event analysis), with cross-modal prediction error as the primary independent variable and standard financial ratios as controls. This tier answers: does the coherence signal correlate with real-world risk?

**Tier 3: Supervised Distress Prediction.** Fin-JEPA embeddings are used as features for downstream classification and regression tasks, benchmarked against the baselines in Section 5.1.4. Classification metrics include AUROC, precision-recall AUC, and calibration curves. This tier answers: does the architecture outperform simpler alternatives for actionable risk prediction?

### **5.1.4 Downstream Prediction Tasks**

To be explicit about what is being predicted: the downstream tasks are binary classification for each of five disaggregated adverse event types, time-to-event regression for distress onset, and continuous risk scoring where the coherence measure is evaluated as a standalone predictor. The five prediction targets are:

* Significant stock price decline (\>20% over 12 months, controlling for market and sector returns)

* Earnings restatement or material misstatement

* Audit opinion qualification or going-concern opinion

* SEC enforcement action

* Bankruptcy filing

Each outcome is evaluated independently. Disaggregating outcomes prevents conflation of market risk with firm-specific risk and allows the signal’s discriminative power to be assessed for each event type. Note that certain categories (audit qualifications, SEC enforcement) will have small sample sizes even in the full corpus; statistical power limitations for these rare events are reported explicitly. Labels are sourced from established databases: UCLA-LoPucki Bankruptcy Research Database for bankruptcy filings, SEC AAER database for enforcement actions, Compustat for restatements, and Audit Analytics or equivalent for audit opinion changes.

### **5.1.5 Baselines**

The following baselines isolate each component’s contribution:

* **Financial Ratios Only:** Logistic regression on raw financial ratios.

* **Simple Divergence Metric:** Correlation between MDA sentiment (Loughran-McDonald) and a financial health composite (Altman Z-score or similar). This is a critical baseline—if a crude sentiment-vs-ratio disagreement captures most of the coherence signal’s predictive power, the architectural contribution is undermined.

* **Market-Based Signals:** Classifier on contemporaneous market data features—price momentum, realised volatility, short interest, abnormal trading volume—computed as of the filing date. This is the most dangerous baseline: if the market has already priced in the risk by the time the 10-K is published (typically 60–90 days after fiscal year end), the coherence signal may simply echo what trading data already reflects.

* **Generic LLM Embeddings:** Classifier on document embeddings from a general-purpose pretrained language model.

* **FinBERT without JEPA:** Classifier on finance-specific document embeddings without the cross-modal objective.

* **Naïve Multimodal Concatenation:** Classifier on concatenated financial and document feature vectors without JEPA latent alignment.

* **XGBoost Leaf Embeddings:** Classifier on dense projections of XGBoost leaf node assignments.

* **NLP Sentiment Baseline:** Classifier combining financial ratios with MDA sentiment scores.

### **5.1.6 Market Signal Redundancy Test**

Regulatory filings are backward-looking documents published with a substantial lag: 10-K annual reports are typically filed 60–90 days after fiscal year end. By the time cross-modal incoherence is detectable in a filing, the market may have already priced in the underlying risk through daily trading activity. If the coherence signal merely confirms what contemporaneous price data already reflects, its practical value is limited.

The market signal redundancy test adds contemporaneous market features (price momentum, realised volatility, short interest, abnormal volume) as controls in the distress prediction regressions. The coherence signal must demonstrate incremental predictive power after conditioning on these market-based signals. If the coherence signal’s coefficient becomes statistically insignificant after adding market controls, the signal is redundant with market pricing and the contribution is substantially weakened. Conversely, if the signal remains predictive after conditioning on market data, it provides information that the market has not yet fully incorporated—a much stronger claim.

### **5.1.7 Financial Encoder Contribution Test**

A subtle failure mode of the cross-modal architecture is that the predictor might learn to generate plausible financial embeddings primarily from text features, effectively ignoring the financial encoder’s input. Language models are inherently information-dense—FinBERT embeddings of MDA text contain implicit information about the company’s financial state (management discussing “liquidity challenges” or “strong cash generation”). If the predictor relies on the language manifold to guess the most likely financial state, prediction error would still correlate with distress (unusual companies produce higher error), but the financial modality would contribute nothing. This failure would be invisible in standard evaluation metrics.

To detect this, the financial encoder’s input is replaced with random noise vectors of matching dimensionality while keeping the document encoder and predictor unchanged. If the coherence signal’s predictive power degrades minimally under noise replacement, the financial modality is being ignored and the architecture is functioning as an elaborate document-only model. Significant degradation confirms that both modalities genuinely contribute to the signal. This is a cheap ablation that should be run early in the evaluation phase, as it determines whether the cross-modal framing is justified.

### **5.1.8 Complexity Confound Analysis**

A dedicated analysis tests whether the coherence signal is primarily a proxy for business complexity rather than genuine narrative-financial inconsistency. Complexity and coherence may be partially overlapping constructs—complex businesses naturally produce both harder-to-parse narratives and more unusual financial profiles, and complexity itself can be a risk factor (e.g., conglomerates or companies with opaque structures). The goal is not to discard the complexity component but to decompose the relative contributions and determine whether coherence has predictive power beyond what complexity alone explains.

The approach has two components:

**Cross-sectional controls.** The coherence signal is re-evaluated after controlling for business complexity measures: number of reported business segments, MDA word count, number of distinct XBRL tags filed, and revenue concentration (Herfindahl index across segments). If the signal’s predictive power substantially diminishes after adding these controls, the complexity-proxy interpretation is supported.

**Within-firm temporal analysis.** Firm-fixed-effects regressions test whether changes in a company’s coherence score over time predict subsequent distress, holding time-invariant firm characteristics constant. A within-firm effect is much harder to explain as a complexity proxy, since business complexity changes slowly while coherence changes driven by narrative-financial divergence can shift rapidly.

### **5.1.9 Directional Decomposition**

The bidirectional prediction error is decomposed into financial-to-document error (the financial encoder’s view cannot predict what the document encoder sees, suggesting narrative inflation) and document-to-financial error (the document encoder’s view cannot predict what the financial encoder sees, suggesting hidden financial risk). Each direction is evaluated independently for predictive power, and interpretability case studies examine high-error observations to determine whether the flagged inconsistencies correspond to identifiable narrative-financial tensions. Note that document-to-financial error need not indicate fraud—it can also capture legitimate forward-looking strategic signals in the narrative that have not yet materialised in the financials.

### **5.1.10 Temporal Evaluation Strategy**

All train/test splits are strictly temporal. The primary split trains on 2012–2020 and evaluates on 2021–2024. To address the concern that this evaluation window spans exclusively extraordinary market conditions (post-COVID recovery, inflation shock, rate hiking cycle), a rolling temporal split is also employed: training on years 1 through T and evaluating on year T+1, repeated across multiple values of T. This produces performance estimates across multiple market regimes, though at significant computational cost.

Survivorship bias is controlled by including all companies that delisted, went bankrupt, or were acquired during the sample period. Sector-stratified evaluation ensures the signal is not simply capturing sector-level effects.

### **5.1.11 Success Criteria**

The coherence signal must be (a) predictive of at least one disaggregated distress outcome at conventional significance levels, (b) additive to the best unimodal baseline, (c) superior to the simple divergence metric, (d) incrementally predictive after conditioning on contemporaneous market signals, and (e) significantly degraded when the financial encoder input is replaced with noise. If it fails any of (a)–(c), the signal concept requires revision. Failure on (d) limits the practical claim to confirmation rather than prediction. Failure on (e) undermines the cross-modal framing entirely.

### **5.1.12 Output**

A full research paper introducing Fin-JEPA, the cross-modal coherence signal, and the suite of validation tests (complexity confound, market redundancy, encoder contribution). Target venues: ACM ICAIF (primary), AAAI Workshop on Financial Technology.

## **5.2 Study 2: Data Efficiency Bounds**

### **5.2.1 Motivation**

Study 2 transforms the question from “does this work?” to “under what conditions does this work?”—a far more useful result for the community. The ablation is the entire focus, not a section competing for space with the architecture description.

### **5.2.2 Ablation Dimensions**

**Company Count.** The model is retrained from scratch at each corpus size: 4,000, 2,000, 500, 100, and 50 companies. The critical thresholds are 100 and 50, corresponding to the approximate sizes of frontier exchanges like the JSE (\~100 companies), Trinidad and Tobago Stock Exchange (\~40), and Barbados Stock Exchange (\~20).

**History Depth.** The training window is shortened from 12+ years to 10, 7, 5, and 3 years. This tests whether the model needs long histories to learn stable cross-modal relationships.

**Trading Data Density.** Market data frequency is thinned from daily to weekly, then monthly, simulating the illiquidity characteristic of frontier markets.

**Combined Ablation.** A subset of the full grid combines reduced company count with reduced history depth, producing a two-dimensional scaling surface that identifies the minimum viable configuration.

### **5.2.3 Sampling Strategy**

A critical methodological choice: at small N, the sector composition of the sample matters. Two sampling strategies are employed at each company count level:

**Sector-preserving sampling.** Companies are drawn to match the US large-cap sector distribution. This isolates the effect of corpus size while holding sector composition constant.

**Frontier-market-weighted sampling.** Companies are drawn with sector weights approximating those of the target frontier market (e.g., JSE-like weights: heavy financials and conglomerates, minimal technology). This tests whether the ablation results hold under the actual distributional shift that frontier market deployment implies. Fifty companies drawn from 4,000 US filers with US sector weights simulate “small N”; fifty companies drawn with JSE-like weights simulate “small N from a different distribution,” which is the actual deployment scenario.

All ablation models are evaluated on the same held-out test set from the full corpus to ensure comparability.

### **5.2.4 Statistical Power**

At small corpus sizes, statistical power is a binding constraint. A 50-company subset with 12 years of history yields approximately 600 company-year observations, of which perhaps 15–30 involve a distress event (depending on definition). This may be insufficient to detect moderate effect sizes. The ablation study reports statistical power analysis at each level, explicitly characterising what can and cannot be inferred. The operational definition of “the signal breaks down” is specified in advance: the point at which the coherence signal’s AUROC confidence interval overlaps with the best non-JEPA baseline’s point estimate.

### **5.2.5 Leveraging Study 1 Findings**

Separating the ablation from the architecture study allows Study 2 to focus resources based on Study 1’s findings. If the directional decomposition reveals that one prediction direction (e.g., financial-to-document) carries the majority of predictive signal, the ablation can focus on that direction, reducing the computational grid. If certain distress outcomes respond more strongly to the coherence signal, the ablation evaluation can prioritise those outcomes.

### **5.2.6 Output**

A research paper focused entirely on data efficiency bounds and the scaling surface. Target venues: ICML (scaling methodology angle), ACM ICAIF, NeurIPS.

## **5.3 Study 3: Frontier Market Validation**

### **5.3.1 Motivation**

Contingent on encouraging ablation results from Study 2, this study validates the approach on JSE data or another frontier market. The study is framed as empirical confirmation of the data efficiency bounds, not as a standalone proof.

### **5.3.2 Data Quality Audit**

Before running any models, a formal data quality audit is conducted as a discrete step. JSE filings are available as PDFs requiring OCR and structured extraction—a fundamentally different data pipeline from SEC XBRL. The audit measures extraction accuracy on a stratified sample of JSE filings across sectors, company sizes, and filing years, quantifying error rates for financial statement line items and MDA text extraction against human-verified ground truth.

If extraction error rates exceed a pre-specified threshold (determined based on the sensitivity analysis from Study 2, which establishes how much input noise the coherence signal tolerates), any subsequent model performance degradation is attributable to pipeline quality rather than method failure. This ensures Study 3 results are interpretable regardless of outcome: poor performance accompanied by high extraction error is a data engineering problem, not a methodological one.

### **5.3.3 Design**

The JSE comprises approximately 100 listed companies across main and junior markets. The study tests whether the approach transfers across regulatory regimes (US GAAP to IFRS), market microstructures, and financial language conventions. A domain-adversarial training component or regional adaptation layer may be employed to align GAAP and IFRS embedding spaces, depending on findings from Study 1 regarding domain adaptation value.

The two-paper separation between architecture (Study 1\) and frontier validation (Study 3\) is strictly stronger than a single frontier-market paper. A negative result on the JSE alone would be uninterpretable—attributable to data quality, extraction errors, or market microstructure rather than the method. With the ablation curve from Study 2 and the data quality audit, a negative result becomes precisely informative: either the JSE falls below the established viability threshold, or there are market-specific factors beyond corpus size that limit transfer, or the extraction pipeline introduces too much noise. Each interpretation points toward a different corrective action.

### **5.3.4 Partnership**

Requires a formal data partnership with the Jamaica Stock Exchange. Aeon Technology Solutions, where the principal investigator serves as Chief Data Officer, provides the industry partnership and data engineering capacity. Academic partners provide research infrastructure and ethics oversight.

### **5.3.5 Output**

A research paper validating frontier market deployment. Target venues: ACM ICAIF, NeurIPS ML4D workshop, ACM DEV, journals focused on AI for developing economies.

# **6\. Data Collection and Preparation**

## **6.1 Primary Dataset: SEC EDGAR**

The primary dataset leverages the SEC’s EDGAR system, which provides structured access to regulatory filings for all US-listed public companies. The EDGAR corpus provides the following for each company-year:

**Narrative documents.** Management Discussion and Analysis (MD\&A) sections from 10-K annual filings and 10-Q quarterly filings, available as pre-parsed text through established EDGAR full-text extraction pipelines. Risk factor disclosures, auditor’s reports, and notes to financial statements are also extractable.

**Structured financials.** Financial statement data in XBRL format, machine-readable and standardised. The SEC has required XBRL tagging since 2009 for large accelerated filers and since 2012 for all filers.

**Market data.** Stock prices, trading volumes, and corporate actions from CRSP, Yahoo Finance, or equivalent. Bankruptcy and distress events from UCLA-LoPucki Bankruptcy Research Database, Compustat, and SEC filings.

## **6.2 Corpus Scope**

| Parameter | Full Corpus | Notes |
| :---- | :---- | :---- |
| Companies | \~4,000+ active filers | All SEC-reporting companies with 10-K filings |
| Temporal coverage | 2012–2024 (XBRL era) | Structured financials consistently available post-2012 |
| Filing types | 10-K (annual) \+ 10-Q (quarterly) | \~16,000+ company-year observations (annual only) |
| MDA availability | Pre-parsed from full-text filings | Established extraction pipelines exist |
| XBRL financials | Machine-readable, standardised | No PDF extraction required |
| Distress events | Disaggregated by type | Multiple established databases |

## **6.3 Future Validation Dataset: Jamaica Stock Exchange (Study 3\)**

Contingent on Study 2 results, a validation study on JSE data will be pursued. The JSE comprises approximately 100 listed companies with regulatory filings available as PDFs requiring extraction. A formal data partnership will be developed in parallel, with no dependency on the Study 0–2 timeline. The data quality audit described in Section 5.3.2 will precede any modelling work.

# **7\. Ethical Considerations**

Studies 0–2 use exclusively public data. SEC filings are mandated public disclosures, XBRL data is explicitly designed for machine consumption, and market data is publicly available. No private, proprietary, or non-public information is used. No institutional review board approval is required.

Study 3, if pursued, will require a data access agreement with the JSE and potentially ethics board approval through the academic partner institution.

The model’s risk assessments are intended as decision-support tools, not automated decision systems. All publications will explicitly discuss limitations and the necessity of human oversight in financial decision-making.

# **8\. Indicative Timeline**

## **8.1 Study 0 Timeline**

| Phase | Key Activities | Duration |
| :---- | :---- | :---- |
| 0.1 | EDGAR data pipeline: XBRL extraction, market data collection | Months 1–2 |
| 0.2 | FT-Transformer implementation, self-supervised pretraining experiments | Months 2–3 |
| 0.3 | Baseline comparison (XGBoost, logistic regression), write-up | Months 3–4 |

## **8.2 Study 1 Timeline**

| Phase | Key Activities | Duration |
| :---- | :---- | :---- |
| 1.1 | MDA parsing pipeline, dataset assembly | Months 3–4 |
| 1.2 | Fin-JEPA architecture implementation | Months 4–6 |
| 1.3 | Full-corpus pretraining and hyperparameter tuning | Months 6–7 |
| 1.4 | Financial encoder contribution test (noise ablation) | Month 7 |
| 1.5 | Core evaluation: coherence signal, baselines, market redundancy test | Months 7–9 |
| 1.6 | Complexity confound analysis, directional decomposition | Months 9–10 |
| 1.7 | Paper writing and submission | Months 10–12 |

## **8.3 Study 2 Timeline**

| Phase | Key Activities | Duration |
| :---- | :---- | :---- |
| 2.1 | Ablation infrastructure: sampling strategies, evaluation harness | Months 10–11 |
| 2.2 | Ablation grid: company count, history depth, trading density | Months 11–14 |
| 2.3 | Frontier-market-weighted sampling variants | Months 14–15 |
| 2.4 | Statistical power analysis, scaling surface construction | Months 15–16 |
| 2.5 | Paper writing and submission | Months 16–18 |

## **8.4 Study 3 Timeline (parallel track, contingent)**

| Phase | Key Activities | Duration |
| :---- | :---- | :---- |
| 3.A | JSE partnership negotiation and data access agreement | Months 1–4 |
| 3.B | JSE data extraction pipeline (PDF to structured data) | Months 4–8 |
| 3.C | Data quality audit: extraction accuracy assessment | Months 8–10 |
| 3.D | Model deployment and evaluation against Study 2 bounds | Months 16–20 |
| 3.E | Paper writing and submission | Months 20–22 |

Note: Studies 0 and 1 overlap in their data pipeline phases. Study 3 partnership development runs in parallel from month 1 with no dependency on the Study 0–2 timeline. The financial encoder contribution test (Phase 1.4) is scheduled early in the evaluation sequence as a go/no-go check on the cross-modal framing.

# **9\. Expected Contributions**

* **Architectural Contribution:** Fin-JEPA as a novel adaptation of Joint Embedding Predictive Architecture to multi-modal financial data, extending the JEPA paradigm beyond vision and video, with explicit justification for the predictive over contrastive paradigm.

* **Signal Contribution:** Cross-modal coherence—the prediction error between narrative and financial encoders—as an empirically validated risk indicator, with explicit testing against a complexity-proxy alternative, contemporaneous market signals, and a noise-replacement control for financial encoder contribution.

* **Scaling Contribution:** Empirically grounded data efficiency bounds with both sector-preserving and frontier-market-weighted ablation, enabling any capital market to assess viability based on its own parameters.

* **Directional Contribution:** Decomposition of the coherence signal into narrative inflation and hidden risk components, with interpretability case studies connecting statistical patterns to identifiable disclosure inconsistencies.

* **Methodological Contribution:** A rigorous incremental study design for evaluating financial ML approaches under data scarcity, with explicit success criteria, go/no-go decision points, and a formal data quality audit protocol for frontier market deployment.

* **Practical Contribution:** Open-source release including model weights, training code, EDGAR extraction pipeline, company universe definitions at each ablation level, distress event labels with sources, and train/test split specifications for full reproducibility.

# **10\. Partnership and Data Access**

**Studies 0–2 (SEC data):** No partnership required. SEC EDGAR is a public system with free access to all filings. XBRL data is available through the SEC’s EDGAR APIs and established academic datasets (Compustat, WRDS). Market data is available from CRSP (academic) or Yahoo Finance (public). Distress events are available from UCLA-LoPucki BRD and SEC AAER databases.

**Study 3 (JSE data):** Requires a formal data partnership with the Jamaica Stock Exchange for bulk access to regulatory filings, historical market data, and company registry information. Aeon Technology Solutions provides the industry partnership and data engineering capacity. Academic partners at CBU or UWI would provide research infrastructure and ethics oversight. The decoupling of Study 3 from Studies 0–2 means the partnership can be developed in parallel with no dependency on the research timeline.

# **11\. Key References**

*LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Technical report, Meta AI.*

*Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023\.*

*Bardes, A., et al. (2024). V-JEPA: The Next Step Toward Advanced Machine Intelligence. Meta AI.*

*Chen, D., et al. (2025). VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language. arXiv:2512.10942.*

*Huang, H., et al. (2025). LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures. arXiv:2509.14252.*

*Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021\.*

*Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.*

*Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. Journal of Finance 66(1).*