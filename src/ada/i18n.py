"""Translation strings for UI + brief.

Two namespaces:
  - `BRIEF[lang]` — section headers and labels for the analytic brief
  - `UI[lang]` — Streamlit UI strings

Default fallback chain: zh-TW → zh → en.
"""
from __future__ import annotations


def _lang_key(language: str | None) -> str:
    if not language:
        return "en"
    if language.startswith("zh"):
        return "zh-TW"  # treat any zh as zh-TW for consistent Traditional script output
    return "en"


# ─────────────────────────────────────────────────────────────────────────────
# Brief rendering strings
# ─────────────────────────────────────────────────────────────────────────────

BRIEF: dict[str, dict[str, str]] = {
    "zh-TW": {
        "title_template": "整合分析簡報 — {project}",
        "field_run_id": "執行 ID",
        "field_project": "專案",
        "field_date": "日期",
        "field_dataset": "資料集",
        "field_rows": "筆",
        "field_capabilities": "可用功能",
        "field_status": "狀態",
        "status_draft": "草稿 — 待同儕審查",

        "section_1": "第一節 — 主要發現",
        "section_2": "第二節 — 情感分布（按主題分列，禁止整體加總）",
        "section_3": "第三節 — 主題 / 敘事登錄表",
        "section_4": "第四節 — 放大行為指標",
        "section_5": "第五節 — 分析限制",
        "section_6": "第六節 — 後續建議",

        "no_findings": "（無發現產出 — 資料不足或上游階段已跳過。）",
        "label_bluf": "BLUF      ",
        "label_evidence": "證據      ",
        "label_confidence": "信心等級  ",
        "label_limitation": "限制",
        "label_recommend": "建議      ",

        "matrix_unavailable": "（主題 × 情感矩陣不可用。）",
        "matrix_warning_1": "⚠ 不可引用整體 'X% 為負面' 的數字。負面情感集中於特定敘事 —",
        "matrix_warning_2": "  必須按敘事分列報告。",

        "matrix_pos": "正面",
        "matrix_neg": "負面",
        "matrix_dst": "強烈負面",
        "matrix_ntr": "中性",
        "matrix_count": "筆數",

        "narrative_topic": "主題",
        "narrative_posts": "篇貼文",
        "narrative_statement": "敘事",
        "narrative_actor": "行為者",
        "narrative_desired": "預期反應",

        "amp_proxy01": "Proxy 01 互動數集中度 : 前 5% 貼文承載 {top5}% 的總互動數",
        "amp_proxy02": "Proxy 02 帳號類型集中度: {n_topics} 個主題單一帳號類型 > 70%",
        "amp_proxy03": "Proxy 03 時序爆發       : {n_bursts} 次超過 {sigma}σ 閾值的爆發",
        "amp_proxy04": "Proxy 04 內容重複度     : {dup_pct}% 重複率",
        "amp_signals":  "協同信號清單           : {triggered}/{total} → 信心等級 {confidence}",

        "lim_no_temporal":  "[資料覆蓋] 無時間戳欄位 — 時序分析（Proxy 03、相位分析）已略過",
        "lim_no_engagement": "[資料覆蓋] 無互動數欄位 — Proxy 01（互動數集中度）已略過",
        "lim_no_author":    "[資料覆蓋] 無帳號類型欄位 — Proxy 02（帳號集中度）已略過",
        "lim_model_sentiment": (
            "[模型] 情感分析使用 Tier 1 + Tier 2 基線（規則 + 詞典，約 120 個 zh-TW 種子詞）；"
            "未使用 transformer 模型。情境相依的微妙情感可能被誤分類。"
        ),
        "lim_model_topic": (
            "[模型] 主題建模使用 BERTopic + 多語言 MiniLM 嵌入；離群率與主題數量"
            "對 min_topic_size 與 nr_topics 敏感 — 公布最終百分比前請透過 "
            "domain_knowledge.thresholds 重新調校。"
        ),
        "lim_causal_proxies": (
            "[因果] 四項放大行為指標皆為「代理」（proxy）而非直接測量。"
            "本資料集缺少追蹤關係圖、跨平台帳號識別與帳號歷史 — "
            "僅憑這些信號無法明確主張存在協同行為。"
        ),
        "lim_causal_topic_sent": (
            "[因果] 主題層情感百分比不能建立因果關係。"
            "高互動群集可能反映演算法推播而非有機共鳴。"
        ),

        "finding_title_neg": "最高風險敘事：{topic}",
        "finding_title_pos": "最強反向敘事：{topic}",
        "finding_title_amp": "放大行為信號：{conf} 信心",

        "bluf_neg_template": (
            "敘事群集「{topic}」在 {posts:,} 篇貼文中承載最高的負面情感濃度（{pct:.1f}%）。"
        ),
        "bluf_pos_template": (
            "敘事群集「{topic}」呈現最高的正面情感佔比（{pct:.1f}%），共 {posts:,} 篇貼文 — "
            "可作為傳播回應的反向敘事素材。"
        ),
        "bluf_amp_template": (
            "{triggered}/{total} 個協同代理指標觸發；前 5% 貼文的互動集中度為 "
            "{top5}%，內容重複率為 {dup}%。"
        ),

        "evidence_neg_template": (
            "主題 × 情感矩陣：此群集 {pct:.1f}% 為負面或強烈負面（與資料集平均比較）。"
            "敘事摘要：{stmt}"
        ),
        "evidence_pos_template": (
            "主題 × 情感矩陣：此群集 {pct:.1f}% 為正面。敘事摘要：{stmt}"
        ),
        "evidence_amp_template": "信號清單：{summary}",

        "conf_reason_neg": (
            "主題與情感信號達 HIGH；敘事框架尚需人工確認"
            "（在無 LLM 支援時為樣板輸出）。"
        ),
        "conf_reason_pos": (
            "正面佔比信號可靠；反向敘事的具體框架需公開使用前先行人工審閱。"
        ),

        "lim_neg_1": "主題層情感 % 反映基線詞典的判讀；微妙與反諷可能被誤分類。",
        "lim_neg_2": (
            "群集定義來自 BERTopic 與多語言 MiniLM 嵌入；"
            "微小群集或合併主題可能改變這些數字。"
        ),
        "lim_pos_1": "正面標籤可能包含未被反諷正規式偵測的內容。",
        "lim_pos_2": "高正面敘事可能包含 hashtag 引致的偏誤（例如 #台灣加油）。",
        "lim_amp_1": (
            "四項代理皆為替代測量 — 缺少追蹤關係圖、跨平台帳號識別與帳號歷史。"
        ),
        "lim_amp_2": (
            "本資料集無法評估帳號層級的時序間隔與跨平台同步性；"
            "無法明確主張存在協同行為。"
        ),

        "rec_neg_template": (
            "將「{topic}」列為首要追蹤的風險敘事。在公開發布前請確認行為者（{actor}）"
            "並審閱敘事陳述。"
        ),
        "rec_pos_template": (
            "以「{topic}」作為任何正面框架傳播的基礎。請手動挑選代表性引言；"
            "切勿自動匯總。"
        ),
        "rec_amp_template": (
            "視為值得後續調查的初步信號，建議與平台信任暨安全團隊合作。"
            "在缺乏帳號層級證據前，不可公開主張存在協同行為。"
        ),

        "brief_end": "簡報結束 — 草稿，待同儕審查",
        "no_actor": "（待人工判定）",
        "narrative_unavailable": "（無敘事提取結果）",

        "confidence_HIGH": "高",
        "confidence_MODERATE": "中度",
        "confidence_LOW": "低",
    },

    "en": {
        "title_template": "Integrated Analytic Brief — {project}",
        "field_run_id": "Run ID",
        "field_project": "Project",
        "field_date": "Date",
        "field_dataset": "Dataset",
        "field_rows": "rows",
        "field_capabilities": "Capabilities used",
        "field_status": "Status",
        "status_draft": "Draft — pending peer review",

        "section_1": "Section 1 — Key findings",
        "section_2": "Section 2 — Sentiment distribution (per topic, never aggregated)",
        "section_3": "Section 3 — Topic / narrative registry",
        "section_4": "Section 4 — Amplification indicators",
        "section_5": "Section 5 — Analytic limitations",
        "section_6": "Section 6 — Recommendations",

        "no_findings": "(No findings produced — insufficient data or skipped upstream stages.)",
        "label_bluf": "BLUF       ",
        "label_evidence": "EVIDENCE   ",
        "label_confidence": "CONFIDENCE ",
        "label_limitation": "LIMITATION",
        "label_recommend": "RECOMMEND  ",

        "matrix_unavailable": "(Topic × sentiment matrix unavailable.)",
        "matrix_warning_1": "⚠ Do not cite an aggregate 'X% negative' figure. Negative sentiment is",
        "matrix_warning_2": "  concentrated in specific narratives — report by narrative.",

        "matrix_pos": "pos",
        "matrix_neg": "neg",
        "matrix_dst": "dst",
        "matrix_ntr": "ntr",
        "matrix_count": "n",

        "narrative_topic": "Topic",
        "narrative_posts": "posts",
        "narrative_statement": "Narrative",
        "narrative_actor": "Actor",
        "narrative_desired": "Desired reaction",

        "amp_proxy01": "Proxy 01 engagement concentration : top 5% carry {top5}% of total engagement",
        "amp_proxy02": "Proxy 02 author-type concentration: {n_topics} topics > 70% single author type",
        "amp_proxy03": "Proxy 03 temporal bursts          : {n_bursts} bursts above {sigma}σ",
        "amp_proxy04": "Proxy 04 content duplication      : {dup_pct}% duplicate rate",
        "amp_signals":  "Coordination signal checklist     : {triggered}/{total} → confidence {confidence}",

        "lim_no_temporal":  "[data coverage] no timestamp column — temporal analyses (Proxy 03, phase analysis) skipped",
        "lim_no_engagement": "[data coverage] no engagement column — Proxy 01 (engagement concentration) skipped",
        "lim_no_author":    "[data coverage] no author column — Proxy 02 (author concentration) skipped",
        "lim_model_sentiment": (
            "[model] Sentiment uses a Tier 1 + Tier 2 baseline (rules + lexicon, ~120 zh-TW seed terms); "
            "no transformer model. Subtle context-dependent sentiment can be misclassified."
        ),
        "lim_model_topic": (
            "[model] Topic clustering uses BERTopic with multilingual MiniLM embeddings; outlier rate "
            "and topic count are sensitive to min_topic_size and nr_topics — re-tune via "
            "domain_knowledge.thresholds before publishing final percentages."
        ),
        "lim_causal_proxies": (
            "[causal] All four amplification indicators are PROXIES, not measurements. The dataset "
            "lacks follower graphs, cross-platform identity, and account history — "
            "coordination cannot be definitively asserted from these signals alone."
        ),
        "lim_causal_topic_sent": (
            "[causal] Per-topic sentiment %s do not establish causation. High-engagement clusters may "
            "reflect algorithmic amplification rather than organic resonance."
        ),

        "finding_title_neg": "Highest-risk narrative: {topic}",
        "finding_title_pos": "Strongest counter-narrative: {topic}",
        "finding_title_amp": "Amplification signal: {conf} confidence",

        "bluf_neg_template": (
            "The narrative cluster '{topic}' carries the highest concentration of "
            "negative sentiment ({pct:.1f}%) across {posts:,} posts."
        ),
        "bluf_pos_template": (
            "The narrative cluster '{topic}' shows the highest positive-sentiment share "
            "({pct:.1f}%) across {posts:,} posts — a candidate for the counter-narrative "
            "section of any communications response."
        ),
        "bluf_amp_template": (
            "{triggered}/{total} coordination proxies fired; engagement concentration is "
            "{top5}% (top 5%) and content duplication is {dup}%."
        ),

        "evidence_neg_template": (
            "Per-topic sentiment matrix: {pct:.1f}% NEGATIVE+DISTRESS in this cluster "
            "vs. dataset average. Narrative summary: {stmt}"
        ),
        "evidence_pos_template": (
            "Per-topic sentiment matrix: {pct:.1f}% POSITIVE in this cluster. "
            "Narrative summary: {stmt}"
        ),
        "evidence_amp_template": "Signal checklist: {summary}",

        "conf_reason_neg": (
            "Topic + sentiment signals are HIGH; narrative framing requires "
            "human confirmation (currently template-based unless LLM was available)."
        ),
        "conf_reason_pos": (
            "POSITIVE share signal is reliable; specific framing of the counter-narrative "
            "requires human review before public use."
        ),

        "lim_neg_1": "Per-topic sentiment %s reflect a baseline lexicon; nuance and irony may be misclassified.",
        "lim_neg_2": (
            "Cluster definitions come from BERTopic with multilingual MiniLM embeddings; "
            "small clusters or merged topics can shift these numbers."
        ),
        "lim_pos_1": "POSITIVE labels can include sarcasm not caught by the sarcasm regex.",
        "lim_pos_2": "High-positive narrative may include hashtag-driven artifacts (e.g. #台灣加油).",
        "lim_amp_1": (
            "All four proxies are surrogates — no follower graph, no cross-platform "
            "identity resolution, no per-account history available."
        ),
        "lim_amp_2": (
            "Account-level timing intervals and cross-platform synchrony cannot be "
            "evaluated with this dataset; coordination cannot be definitively asserted."
        ),

        "rec_neg_template": (
            "Treat '{topic}' as the primary risk narrative for follow-up. "
            "Verify the actor ({actor}) and review the narrative statement before publishing."
        ),
        "rec_pos_template": (
            "Use '{topic}' as the foundation for any positive-framing communications. "
            "Hand-pick representative quotes; do not auto-aggregate."
        ),
        "rec_amp_template": (
            "Treat as a preliminary signal worth investigating with platform "
            "trust-and-safety teams. Do not act publicly on coordination claims "
            "without account-level evidence."
        ),

        "brief_end": "Brief end — draft, awaiting peer review",
        "no_actor": "(awaiting human judgment)",
        "narrative_unavailable": "(no narrative extracted)",

        "confidence_HIGH": "HIGH",
        "confidence_MODERATE": "MODERATE",
        "confidence_LOW": "LOW",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# UI strings
# ─────────────────────────────────────────────────────────────────────────────

UI: dict[str, dict[str, str]] = {
    "zh-TW": {
        "app_title": "Agentic Data Analyst — 情感分析智能助理",
        "app_subtitle": "資料攝入 → 情感 → 主題 → 敘事 → 整合分析簡報",

        # Sidebar
        "nav_start": "① 啟動分析",
        "nav_progress": "② 進度追蹤",
        "nav_hitl": "③ 人工確認",
        "nav_report": "④ 分析報告",
        "current_run_label": "目前執行",
        "no_active_run": "（尚未啟動）",

        # Start page
        "start_header": "啟動新的分析",
        "start_intro": (
            "上傳社群媒體資料集（CSV / XLSX / JSON），代理人會自動執行資料攝入、"
            "EDA、清理、情感分析、主題建模、敘事抽取與整合分析簡報。"
            "在關鍵節點會請您確認決策。"
        ),
        "start_file_label": "資料檔案",
        "start_file_help": "支援 .csv / .xlsx / .json / .jsonl / .parquet",
        "start_project_label": "專案名稱",
        "start_project_help": "用於組織輸出檔案；同名專案會共用領域知識（domain.yaml）",
        "start_prompt_label": "額外說明（可選）",
        "start_prompt_help": "提供分析目標或資料背景，協助代理人做出更好的決策",
        "start_run_button": "🚀 開始分析",
        "start_no_file": "請先選擇資料檔案。",

        # Progress page
        "progress_header": "進度追蹤",
        "progress_no_run": "尚無執行中的分析。請先到「啟動分析」頁面開始。",
        "progress_running": "管線執行中...",
        "progress_paused": "⏸ 等待人工確認 — 請至「人工確認」頁面",
        "progress_done": "✅ 分析完成 — 請至「分析報告」頁面查看結果",
        "progress_stages": "管線階段",
        "progress_audit": "稽核日誌",
        "progress_artifacts": "已產出的輸出物",
        "audit_col_time": "時間",
        "audit_col_stage": "階段",
        "audit_col_action": "動作",
        "audit_col_rows": "影響筆數",
        "audit_col_reason": "原因",

        # Stage names
        "stage_ingest": "資料攝入",
        "stage_schema_infer": "綱要推斷",
        "stage_reshape": "資料重塑",
        "stage_eda": "探索分析",
        "stage_clean": "清理 / 三流分割",
        "stage_preprocess": "文字前處理",
        "stage_sentiment": "情感分析",
        "stage_topic": "主題建模",
        "stage_narrative": "敘事抽取",
        "stage_amplification": "放大行為偵測",
        "stage_brief": "整合分析簡報",

        # HITL page
        "hitl_header": "人工確認",
        "hitl_no_question": "目前沒有等待回應的問題。",
        "hitl_why": "為什麼問這題",
        "hitl_submit": "✅ 確認並繼續",
        "hitl_reject": "✗ 駁回（重新推斷）",

        # Schema confirm
        "hitl_schema_title": "確認資料綱要",
        "hitl_schema_intro": (
            "代理人推斷以下欄位對應。請確認或修改 — 這是後續所有分析的基礎。"
        ),
        "hitl_schema_role_id": "ID 欄位",
        "hitl_schema_role_text": "文字欄位（必要）",
        "hitl_schema_role_ts": "時間戳欄位",
        "hitl_schema_role_author": "帳號類型欄位",
        "hitl_schema_role_engagement": "互動數欄位",
        "hitl_schema_role_platform": "平台欄位",
        "hitl_schema_language": "語言",
        "hitl_schema_none": "（無）",
        "hitl_schema_samples_caption": "前 5 筆樣本",

        # Topic label HITL
        "hitl_topic_title": "為主題群集命名",
        "hitl_topic_intro": (
            "BERTopic 找到了 {n} 個主題群集。代理人根據關鍵詞起草了標籤 — "
            "請參考代表性貼文後修改。**這是整個管線中最關鍵的人工輸入：**"
            "標籤會直接出現在最終的整合分析簡報中。"
        ),
        "hitl_topic_cluster": "群集",
        "hitl_topic_size": "篇貼文",
        "hitl_topic_keywords": "關鍵詞",
        "hitl_topic_label_input": "您的標籤",
        "hitl_topic_samples": "代表性貼文（依互動數排序）",
        "hitl_topic_save_to_memory": "📌 將這些標籤加入領域記憶（適用於未來相似資料集）",

        # Sentiment calibrate
        "hitl_sentiment_title": "校正情感標籤",
        "hitl_sentiment_intro": (
            "規則與詞典標註結果分歧較大。請手動標記以下 {n} 筆貼文，"
            "並提示我可能遺漏的文化或反諷模式。"
        ),
        "hitl_sentiment_label_choices": ["POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE-DISTRESS", "UNCERTAIN"],
        "hitl_sentiment_notes": "額外備註（例如「請加入這個反諷模式」）",

        # Report page
        "report_header": "分析報告",
        "report_no_brief": "尚未產出簡報。請完成所有人工確認後再回來查看。",
        "report_download": "📥 下載簡報文字檔",
        "report_findings": "主要發現",
        "report_matrix": "主題 × 情感矩陣",
        "report_figures": "EDA 圖表",
        "report_full_text": "完整簡報",
        "report_run_meta": "執行資訊",
    },

    "en": {
        "app_title": "Agentic Data Analyst — Sentiment Analysis Assistant",
        "app_subtitle": "Ingest → Sentiment → Topic → Narrative → Integrated Brief",
        "nav_start": "① Start",
        "nav_progress": "② Progress",
        "nav_hitl": "③ Human Input",
        "nav_report": "④ Report",
        "current_run_label": "Current run",
        "no_active_run": "(none)",
        "start_header": "Start a new analysis",
        "start_intro": (
            "Upload a tabular dataset (CSV / XLSX / JSON). The agent will run "
            "ingestion, EDA, cleaning, sentiment analysis, topic modeling, "
            "narrative extraction, and produce an integrated brief, asking for "
            "your input at key decision points."
        ),
        "start_file_label": "Dataset file",
        "start_file_help": "Supports .csv / .xlsx / .json / .jsonl / .parquet",
        "start_project_label": "Project name",
        "start_project_help": "Output directory; same-name projects share domain memory",
        "start_prompt_label": "Optional context",
        "start_prompt_help": "Tell the agent about your analysis goal or dataset background",
        "start_run_button": "🚀 Start analysis",
        "start_no_file": "Please choose a dataset file first.",
        "progress_header": "Progress",
        "progress_no_run": "No active run. Start one from the Start page.",
        "progress_running": "Running…",
        "progress_paused": "⏸ Awaiting human input — go to Human Input page",
        "progress_done": "✅ Done — go to Report page",
        "progress_stages": "Stages",
        "progress_audit": "Audit log",
        "progress_artifacts": "Artifacts produced",
        "audit_col_time": "Time",
        "audit_col_stage": "Stage",
        "audit_col_action": "Action",
        "audit_col_rows": "Rows",
        "audit_col_reason": "Reason",
        "stage_ingest": "Ingest",
        "stage_schema_infer": "Schema inference",
        "stage_reshape": "Reshape",
        "stage_eda": "EDA",
        "stage_clean": "Clean / 3-stream split",
        "stage_preprocess": "Preprocess",
        "stage_sentiment": "Sentiment",
        "stage_topic": "Topic modeling",
        "stage_narrative": "Narrative extraction",
        "stage_amplification": "Amplification proxies",
        "stage_brief": "Integrated brief",
        "hitl_header": "Human Input",
        "hitl_no_question": "No questions pending.",
        "hitl_why": "Why I'm asking",
        "hitl_submit": "✅ Confirm and continue",
        "hitl_reject": "✗ Reject",
        "hitl_schema_title": "Confirm dataset schema",
        "hitl_schema_intro": "I inferred this column mapping. Confirm or correct.",
        "hitl_schema_role_id": "ID column",
        "hitl_schema_role_text": "Text column (required)",
        "hitl_schema_role_ts": "Timestamp column",
        "hitl_schema_role_author": "Author column",
        "hitl_schema_role_engagement": "Engagement column",
        "hitl_schema_role_platform": "Platform column",
        "hitl_schema_language": "Language",
        "hitl_schema_none": "(none)",
        "hitl_schema_samples_caption": "First 5 samples",
        "hitl_topic_title": "Label the topic clusters",
        "hitl_topic_intro": (
            "BERTopic found {n} clusters. I drafted labels from top keywords — "
            "please refine based on the sample posts. **This is the most consequential "
            "human input in the pipeline:** labels propagate verbatim into the brief."
        ),
        "hitl_topic_cluster": "Cluster",
        "hitl_topic_size": "posts",
        "hitl_topic_keywords": "Keywords",
        "hitl_topic_label_input": "Your label",
        "hitl_topic_samples": "Representative posts (by engagement)",
        "hitl_topic_save_to_memory": "📌 Save these labels to domain memory (for future similar datasets)",
        "hitl_sentiment_title": "Calibrate sentiment labels",
        "hitl_sentiment_intro": (
            "My rule-based and lexicon-based labels disagree substantially. "
            "Please label these {n} posts and tell me what cultural/sarcasm patterns I may be missing."
        ),
        "hitl_sentiment_label_choices": ["POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE-DISTRESS", "UNCERTAIN"],
        "hitl_sentiment_notes": "Additional notes (e.g. 'add this sarcasm pattern')",
        "report_header": "Analytic report",
        "report_no_brief": "No brief produced yet. Complete the human input steps first.",
        "report_download": "📥 Download brief (.txt)",
        "report_findings": "Key findings",
        "report_matrix": "Topic × sentiment matrix",
        "report_figures": "EDA figures",
        "report_full_text": "Full brief",
        "report_run_meta": "Run info",
    },
}


def t_brief(language: str | None, key: str, **fmt) -> str:
    table = BRIEF[_lang_key(language)]
    s = table.get(key, BRIEF["en"].get(key, key))
    return s.format(**fmt) if fmt else s


def t_ui(language: str | None, key: str, **fmt) -> str:
    table = UI[_lang_key(language)]
    val = table.get(key, UI["en"].get(key, key))
    if isinstance(val, str) and fmt:
        return val.format(**fmt)
    return val


def stage_label(language: str | None, stage_value: str) -> str:
    return t_ui(language, f"stage_{stage_value}")
