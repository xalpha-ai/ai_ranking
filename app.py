from flask import Flask, render_template, request, jsonify, send_file
import re
import json
import io
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from dotenv import load_dotenv

# Import database functions
from database import (
    init_db, save_visibility_score, save_competitive_analysis,
    save_ranking_analysis, save_geographic_score,
    get_historical_visibility_scores, get_latest_scores, get_all_brands
)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global error handlers to ensure JSON responses
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    print(f"Unhandled exception: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': 'An unexpected error occurred', 'message': str(e)}), 500

# Initialize database on startup (non-critical - app works without it)
print("=== Initializing Application ===")
try:
    db_engine = init_db()
    if db_engine:
        print("✓ Database initialized successfully")
    else:
        print("⚠️ Database initialization failed - app will work without historical data")
except Exception as e:
    print(f"⚠️ Database initialization failed: {e}")
    print("⚠️ App will continue without database - analysis will still work!")
    import traceback
    traceback.print_exc()

# Core Data Structures
@dataclass
class VisibilityWeights:
    presence: float = 0.30
    prominence: float = 0.25
    narrative: float = 0.25
    authority: float = 0.20

    def validate(self) -> None:
        total = self.presence + self.prominence + self.narrative + self.authority
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1.0. Got {total}")


@dataclass
class NarrativeRules:
    required_phrases: List[str]
    optional_phrases: List[str]
    forbidden_phrases: List[str]


@dataclass
class BrandConfig:
    brand: str
    industry: str
    question_set: List[str]
    aliases: List[str]
    narrative_required: List[str]
    narrative_optional: List[str]
    narrative_forbidden: List[str]
    official_urls_hint: List[str]


# Text Helpers
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _contains(text: str, phrase: str) -> bool:
    return _normalize(phrase) in _normalize(text)


def _brand_mentioned(answer: str, brand: str, aliases: Optional[List[str]] = None) -> bool:
    a = _normalize(answer)
    variants = [brand] + (aliases or [])
    return any(_normalize(v) in a for v in variants)


def _first_mention_index(answer: str, brand: str, aliases: Optional[List[str]] = None) -> Optional[int]:
    a = _normalize(answer)
    variants = [brand] + (aliases or [])
    hits = []
    for v in variants:
        idx = a.find(_normalize(v))
        if idx != -1:
            hits.append(idx)
    return min(hits) if hits else None


# Scoring Functions
def _prominence_score(answer: str, brand: str, aliases: Optional[List[str]] = None) -> float:
    if not (answer or "").strip():
        return 0.0
    idx = _first_mention_index(answer, brand, aliases)
    if idx is None:
        return 0.0
    frac = idx / max(len(_normalize(answer)), 1)
    if frac <= 0.10:
        return 1.0
    elif frac <= 0.30:
        return 0.7
    elif frac <= 0.60:
        return 0.4
    else:
        return 0.2


def _authority_score(answer: str, official_urls: List[str]) -> float:
    a = _normalize(answer)
    if not official_urls:
        return 0.0
    hits = 0
    for url in official_urls:
        u = _normalize(url)
        domain = re.sub(r"^https?://", "", u).split("/")[0]
        if u and u in a:
            hits += 1
        elif domain and domain in a:
            hits += 1
    if hits == 0:
        return 0.0
    if hits == 1:
        return 0.6
    return 1.0


def _narrative_score(answer: str, rules: NarrativeRules) -> float:
    if not (answer or "").strip():
        return 0.0
    req = rules.required_phrases or []
    opt = rules.optional_phrases or []
    forb = rules.forbidden_phrases or []
    req_hits = sum(1 for p in req if _contains(answer, p))
    opt_hits = sum(1 for p in opt if _contains(answer, p))
    forb_hits = sum(1 for p in forb if _contains(answer, p))
    req_ratio = req_hits / max(len(req), 1)
    score = 0.2 + 0.8 * req_ratio
    score += 0.05 * min(opt_hits, 4)
    score -= 0.30 * forb_hits
    return max(0.0, min(1.0, score))


# Visibility Score Calculator
def calculate_visibility_score(
    responses: List[Dict[str, Any]],
    brand: str,
    official_urls: List[str],
    narrative_rules: NarrativeRules,
    brand_aliases: Optional[List[str]] = None,
    weights: VisibilityWeights = VisibilityWeights(),
) -> Dict[str, Any]:
    weights.validate()
    per = []
    for r in responses:
        ans = r.get("answer", "") or ""
        mentioned = _brand_mentioned(ans, brand, brand_aliases)
        presence = 1.0 if mentioned else 0.0
        prominence = _prominence_score(ans, brand, brand_aliases) if mentioned else 0.0
        narrative = _narrative_score(ans, narrative_rules) if mentioned else 0.0
        authority = _authority_score(ans, official_urls) if mentioned else 0.0
        per.append({
            "question": r.get("question"),
            "model": r.get("model"),
            "presence": presence,
            "prominence": prominence,
            "narrative": narrative,
            "authority": authority,
        })

    def avg(key: str) -> float:
        return sum(x[key] for x in per) / max(len(per), 1)

    presence_avg   = avg("presence")
    prominence_avg = avg("prominence")
    narrative_avg  = avg("narrative")
    authority_avg  = avg("authority")
    final_0_1 = (
        presence_avg   * weights.presence +
        prominence_avg * weights.prominence +
        narrative_avg  * weights.narrative +
        authority_avg  * weights.authority
    )
    visibility_score = round(final_0_1 * 100, 2)
    component_scores = {
        "presence": round(presence_avg * 100, 2),
        "prominence": round(prominence_avg * 100, 2),
        "narrative": round(narrative_avg * 100, 2),
        "authority": round(authority_avg * 100, 2),
    }
    return {
        "visibility_score": visibility_score,
        "component_scores": component_scores,
        "details": per,
        "weights": weights.__dict__,
    }


# AI Config Generator
def _extract_json_from_text(text: str) -> str:
    """
    Clean up model output so it's valid JSON:
    - Strip code fences ```json ... ```
    - Trim leading/trailing junk
    - Slice from first '{' or '[' to last '}' or ']'
    """
    if not text:
        return ""

    cleaned = text.strip()

    # If it's in a ```json ... ``` block, strip fences
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                lines = lines[:i]
                break
        cleaned = "\n".join(lines).strip()

    # Determine if it's an array or object by checking what comes first
    first_bracket = cleaned.find("[")
    first_brace = cleaned.find("{")

    # If array comes before object (or no object found), extract array
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        end_bracket = cleaned.rfind("]")
        if end_bracket > first_bracket:
            return cleaned[first_bracket:end_bracket+1].strip()

    # Otherwise extract object
    if first_brace != -1:
        end_brace = cleaned.rfind("}")
        if end_brace > first_brace:
            return cleaned[first_brace:end_brace+1].strip()

    return cleaned


def generate_brand_config_via_ai(
    brand: str,
    industry: str,
    brand_description: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> BrandConfig:
    prompt = f"""
You are helping define an AI visibility audit for a brand.

BRAND: {brand}
INDUSTRY: {industry}

BRAND DESCRIPTION (free text from the team):
{brand_description}

TASK:
Design a config for measuring AI visibility for this brand.

1) Create a QUESTION SET (10–15 questions) that:
   - A typical buyer or researcher might ask about this industry and this brand
   - Mix of:
     * "best of" questions (top tools / leading providers)
     * comparison questions
     * capability questions
     * "where can I learn more" questions

2) Define NARRATIVE RULES:
   - narrative_required: 3–8 key concepts that MUST appear in a good description of this brand
                         (category, outcomes, core capabilities, positioning)
   - narrative_optional: 5–10 nice-to-have concepts that strengthen the narrative
   - narrative_forbidden: 3–10 things that are clearly wrong / off-category for this brand
                          (wrong industry, wrong product type, common misconceptions)

3) Provide ALIASES:
   - plausible ways people might refer to the brand (short names, spacing variants, etc.)

4) Provide a rough guess of OFFICIAL_URLS_HINT:
   - 1–3 plausible URL patterns, e.g. main domain, /product, /ai

Return STRICT JSON only, matching this schema exactly:

{{
  "brand": "{brand}",
  "industry": "{industry}",
  "question_set": ["..."],
  "aliases": ["..."],
  "narrative_required": ["..."],
  "narrative_optional": ["..."],
  "narrative_forbidden": ["..."],
  "official_urls_hint": ["..."]
}}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = resp.choices[0].message.content
    cleaned = _extract_json_from_text(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("RAW MODEL OUTPUT (first 800 chars):")
        print(raw_text[:800])
        print("\nCLEANED FOR JSON (first 800 chars):")
        print(cleaned[:800])
        raise

    return BrandConfig(
        brand=data["brand"],
        industry=data["industry"],
        question_set=data["question_set"],
        aliases=data.get("aliases", []),
        narrative_required=data.get("narrative_required", []),
        narrative_optional=data.get("narrative_optional", []),
        narrative_forbidden=data.get("narrative_forbidden", []),
        official_urls_hint=data.get("official_urls_hint", []),
    )


def ask_model_for_brand_answers(
    client: OpenAI,
    model_name: str,
    questions: List[str],
) -> List[Dict[str, str]]:
    answers = []
    for q in questions:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": q}],
        )
        answers.append({
            "question": q,
            "model": model_name,
            "answer": resp.choices[0].message.content,
        })
    return answers


def run_visibility_audit_for_brand(
    brand: str,
    industry: str,
    brand_description: str,
    api_key: str,
    scoring_model: str = "gpt-4o-mini",
    answering_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key, timeout=240.0)  # 4 minute timeout for API calls

    config = generate_brand_config_via_ai(
        brand=brand,
        industry=industry,
        brand_description=brand_description,
        client=client,
        model=scoring_model,
    )

    responses = ask_model_for_brand_answers(
        client=client,
        model_name=answering_model,
        questions=config.question_set,
    )

    rules = NarrativeRules(
        required_phrases=config.narrative_required,
        optional_phrases=config.narrative_optional,
        forbidden_phrases=config.narrative_forbidden,
    )
    official_urls = config.official_urls_hint
    vis = calculate_visibility_score(
        responses=responses,
        brand=brand,
        brand_aliases=config.aliases,
        official_urls=official_urls,
        narrative_rules=rules,
    )

    return {
        "brand": brand,
        "industry": industry,
        "config": config,
        "visibility": vis,
        "raw_responses": responses,
    }


def generate_visibility_report_markdown(result: Dict[str, Any]) -> str:
    brand = result["brand"]
    industry = result["industry"]
    config: BrandConfig = result["config"]
    vis = result["visibility"]
    responses = result["raw_responses"]
    score = vis["visibility_score"]
    comps = vis["component_scores"]

    lines = []
    lines.append(f"# AI Visibility Report – {brand}")
    lines.append("")
    lines.append(f"**Industry:** {industry}")
    lines.append("")
    lines.append(f"**Overall Visibility Score:** `{score} / 100`")
    lines.append("")
    lines.append("## Component Scores")
    lines.append("")
    lines.append("| Component   | Score |")
    lines.append("|------------|-------|")
    for k in ["presence", "prominence", "narrative", "authority"]:
        lines.append(f"| {k.capitalize()} | {comps[k]} |")
    lines.append("")
    lines.append("## Brand Config (Auto-Generated)")
    lines.append("")
    lines.append("**Aliases:** " + (", ".join(config.aliases) if config.aliases else "_none_"))
    lines.append("")
    if config.official_urls_hint:
        lines.append("**Official URL Hints (from AI):**")
        for url in config.official_urls_hint:
            lines.append(f"- {url}")
    else:
        lines.append("**Official URL Hints:** _none_")
    lines.append("")
    lines.append("### Narrative – Required Phrases")
    lines.append("")
    if config.narrative_required:
        for p in config.narrative_required:
            lines.append(f"- {p}")
    else:
        lines.append("- _none_")
    lines.append("")
    lines.append("### Narrative – Optional Phrases")
    lines.append("")
    if config.narrative_optional:
        for p in config.narrative_optional:
            lines.append(f"- {p}")
    else:
        lines.append("- _none_")
    lines.append("")
    lines.append("### Narrative – Forbidden Phrases")
    lines.append("")
    if config.narrative_forbidden:
        for p in config.narrative_forbidden:
            lines.append(f"- {p}")
    else:
        lines.append("- _none_")
    lines.append("")
    lines.append("## Question Set Used")
    lines.append("")
    for i, q in enumerate(config.question_set, start=1):
        lines.append(f"{i}. {q}")
    lines.append("")
    lines.append("## Per-Question Summary")
    lines.append("")
    lines.append("| # | Question | Model | Mentioned Brand? |")
    lines.append("|---|----------|-------|------------------|")
    for i, (r, d) in enumerate(zip(responses, vis["details"]), start=1):
        mentioned = "✅" if d["presence"] > 0 else "❌"
        q_short = r["question"].replace("|", "\\|")
        lines.append(f"| {i} | {q_short} | {r['model']} | {mentioned} |")
    lines.append("")
    lines.append("> Tip: Use the narrative rules plus low-scoring components to decide what to fix on your product and info pages.")
    lines.append("")
    return "\n".join(lines)


def generate_ranking_prompts(
    brand: str,
    industry: str,
    brand_description: str,
    client: OpenAI,
    num_prompts: int = 10,
    model: str = "gpt-4o-mini",
) -> List[str]:
    """Generate industry-specific ranking prompts to test brand visibility."""

    prompt = f"""
You are helping create ranking queries to test brand visibility.

BRAND: {brand}
INDUSTRY: {industry}
BRAND DESCRIPTION: {brand_description}

TASK:
Generate {num_prompts} ranking-style questions that a customer might ask when researching products/services in this industry.

These should be questions that would naturally produce a ranked list or "top N" style answer. Mix different categories:
- Best overall products/brands in the industry
- Top sellers or most popular
- Best for specific use cases (e.g., "best family car", "best budget option")
- Best for specific features/capabilities
- Top rated by specific criteria (safety, performance, value, etc.)

Format each as a question that would elicit a ranked list response.

Return STRICT JSON only, as an array of question strings:

[
  "What are the top 10 best-selling cars in 2024?",
  "Which are the best family SUVs for safety?",
  ...
]
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = resp.choices[0].message.content
    cleaned = _extract_json_from_text(raw_text)

    try:
        prompts = json.loads(cleaned)
        return prompts if isinstance(prompts, list) else []
    except json.JSONDecodeError:
        print("Failed to parse ranking prompts JSON")
        print("RAW OUTPUT:", raw_text[:500])
        return []


def extract_brand_ranking(
    response_text: str,
    brand: str,
    brand_aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract where the brand appears in a ranking response."""

    variants = [brand] + (brand_aliases or [])
    normalized_variants = [_normalize(v) for v in variants]

    # Split into lines and look for mentions
    lines = response_text.split('\n')

    position = None
    total_items = 0
    mentioned = False

    for i, line in enumerate(lines, 1):
        normalized_line = _normalize(line)

        # Check if brand is mentioned
        if any(v in normalized_line for v in normalized_variants):
            mentioned = True
            # Try to extract position number
            # Look for patterns like "1.", "#1", "1)", "Top 1:", etc.
            import re
            patterns = [
                r'^\s*(\d+)[\.\)\:]',  # 1. or 1) or 1:
                r'#(\d+)',              # #1
                r'top\s+(\d+)',         # top 1
                r'^(\d+)\s*[-–—]',     # 1 - or 1 – or 1 —
            ]
            for pattern in patterns:
                match = re.search(pattern, normalized_line, re.IGNORECASE)
                if match:
                    position = int(match.group(1))
                    break

            if position is None:
                # If no explicit number, use line number as position
                position = i
            break

    # Try to estimate total items in the list
    for line in lines:
        if re.search(r'^\s*(\d+)[\.\)\:]', line):
            total_items += 1

    if total_items == 0 and mentioned:
        total_items = 10  # Default assumption

    return {
        'mentioned': mentioned,
        'position': position,
        'total_items': total_items,
        'appears_in_top_3': position is not None and position <= 3,
        'appears_in_top_5': position is not None and position <= 5,
        'appears_in_top_10': position is not None and position <= 10,
    }


def analyze_ranking_performance(
    brand: str,
    industry: str,
    brand_description: str,
    brand_aliases: Optional[List[str]] = None,
    client: OpenAI = None,
    num_prompts: int = 10,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Analyze brand ranking performance across multiple ranking queries."""

    # Generate ranking prompts
    ranking_prompts = generate_ranking_prompts(
        brand=brand,
        industry=industry,
        brand_description=brand_description,
        client=client,
        num_prompts=num_prompts,
    )

    results = []

    for prompt_text in ranking_prompts:
        # Ask the ranking question
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
        )

        answer = resp.choices[0].message.content

        # Extract ranking
        ranking_info = extract_brand_ranking(answer, brand, brand_aliases)

        results.append({
            'prompt': prompt_text,
            'answer': answer,
            'mentioned': ranking_info['mentioned'],
            'position': ranking_info['position'],
            'total_items': ranking_info['total_items'],
            'in_top_3': ranking_info['appears_in_top_3'],
            'in_top_5': ranking_info['appears_in_top_5'],
            'in_top_10': ranking_info['appears_in_top_10'],
        })

    # Categorize into strong and weak areas
    strong_areas = []
    weak_areas = []
    not_mentioned = []

    for r in results:
        if not r['mentioned']:
            not_mentioned.append(r)
        elif r['in_top_3']:
            strong_areas.append(r)
        elif r['mentioned'] and (r['position'] is None or r['position'] > 5):
            weak_areas.append(r)
        else:
            # Middle ground - mentioned in top 5 but not top 3
            strong_areas.append(r)

    # Calculate statistics
    total_prompts = len(results)
    mentioned_count = sum(1 for r in results if r['mentioned'])
    top_3_count = sum(1 for r in results if r.get('in_top_3', False))
    top_5_count = sum(1 for r in results if r.get('in_top_5', False))
    top_10_count = sum(1 for r in results if r.get('in_top_10', False))

    avg_position = None
    positions = [r['position'] for r in results if r['position'] is not None]
    if positions:
        avg_position = round(sum(positions) / len(positions), 2)

    return {
        'total_prompts': total_prompts,
        'mentioned_count': mentioned_count,
        'mention_rate': round(mentioned_count / max(total_prompts, 1) * 100, 2),
        'top_3_count': top_3_count,
        'top_5_count': top_5_count,
        'top_10_count': top_10_count,
        'average_position': avg_position,
        'strong_areas': strong_areas,
        'weak_areas': weak_areas,
        'not_mentioned': not_mentioned,
        'all_results': results,
    }


def identify_competitors(
    brand: str,
    industry: str,
    brand_description: str,
    client: OpenAI,
    num_competitors: int = 5,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, str]]:
    """Identify main competitors for a brand using AI."""

    prompt = f"""
You are a business analyst helping identify competitors for a brand.

BRAND: {brand}
INDUSTRY: {industry}

BRAND DESCRIPTION:
{brand_description}

TASK:
Identify the top {num_competitors} main competitors for this brand in the {industry} industry.

For each competitor, provide:
1. Company name
2. Brief description (1-2 sentences about what they do)
3. Why they are a competitor (how they compete with {brand})

Return STRICT JSON only, as an array of competitor objects:

[
  {{
    "name": "Competitor Name",
    "description": "Brief description of what they do...",
    "competitive_reason": "Why they compete with {brand}..."
  }},
  ...
]
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = resp.choices[0].message.content
    cleaned = _extract_json_from_text(raw_text)

    try:
        competitors = json.loads(cleaned)
        return competitors if isinstance(competitors, list) else []
    except json.JSONDecodeError:
        print("Failed to parse competitors JSON")
        print("RAW OUTPUT:", raw_text[:500])
        return []


def run_competitive_analysis(
    brand: str,
    industry: str,
    brand_description: str,
    api_key: str,
    num_competitors: int = 3,
) -> Dict[str, Any]:
    """Run visibility analysis for target brand and competitors, then compare."""

    client = OpenAI(api_key=api_key, timeout=240.0)  # 4 minute timeout for API calls

    # Step 1: Identify competitors
    competitors = identify_competitors(
        brand=brand,
        industry=industry,
        brand_description=brand_description,
        client=client,
        num_competitors=num_competitors,
    )

    # Step 2: Run visibility audit for target brand
    target_result = run_visibility_audit_for_brand(
        brand=brand,
        industry=industry,
        brand_description=brand_description,
        api_key=api_key,
    )

    # Step 3: Run visibility audits for each competitor
    competitor_results = []
    for comp in competitors:
        comp_result = run_visibility_audit_for_brand(
            brand=comp['name'],
            industry=industry,
            brand_description=comp['description'],
            api_key=api_key,
        )
        competitor_results.append({
            'name': comp['name'],
            'description': comp['description'],
            'competitive_reason': comp['competitive_reason'],
            'visibility': comp_result['visibility'],
            'raw_responses': comp_result['raw_responses'],
            'config': comp_result['config'],
        })

    # Step 4: Perform comparative analysis
    analysis = analyze_competitive_position(target_result, competitor_results)

    return {
        'target': target_result,
        'competitors': competitor_results,
        'analysis': analysis,
    }


def analyze_competitive_position(
    target_result: Dict[str, Any],
    competitor_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze target brand's competitive position and identify strengths/weaknesses."""

    target_score = target_result['visibility']['visibility_score']
    target_components = target_result['visibility']['component_scores']
    target_details = target_result['visibility']['details']

    # Gather competitor scores
    competitor_scores = []
    for comp in competitor_results:
        competitor_scores.append({
            'name': comp['name'],
            'score': comp['visibility']['visibility_score'],
            'components': comp['visibility']['component_scores'],
        })

    # Calculate rankings
    all_scores = [{'name': target_result['brand'], 'score': target_score, 'is_target': True}]
    all_scores.extend([{'name': c['name'], 'score': c['score'], 'is_target': False} for c in competitor_scores])
    all_scores.sort(key=lambda x: x['score'], reverse=True)

    target_rank = next(i+1 for i, s in enumerate(all_scores) if s['is_target'])

    # Identify strengths (questions where target excels)
    strengths = []
    weaknesses = []

    for idx, detail in enumerate(target_details):
        question = detail['question']
        target_presence = detail['presence']
        target_prominence = detail['prominence']
        target_narrative = detail['narrative']

        # Calculate average score for this question
        question_score = (target_presence + target_prominence + target_narrative) / 3

        if target_presence == 1.0 and question_score >= 0.7:
            strengths.append({
                'question': question,
                'score': round(question_score * 100, 2),
                'presence': target_presence,
                'prominence': target_prominence,
                'narrative': target_narrative,
            })
        elif target_presence < 0.5 or question_score < 0.4:
            weaknesses.append({
                'question': question,
                'score': round(question_score * 100, 2),
                'presence': target_presence,
                'prominence': target_prominence,
                'narrative': target_narrative,
            })

    # Sort by score
    strengths.sort(key=lambda x: x['score'], reverse=True)
    weaknesses.sort(key=lambda x: x['score'])

    # Component comparison
    component_comparison = {}
    for component in ['presence', 'prominence', 'narrative', 'authority']:
        target_comp_score = target_components[component]
        competitor_avg = sum(c['components'][component] for c in competitor_scores) / max(len(competitor_scores), 1)

        component_comparison[component] = {
            'target': target_comp_score,
            'competitor_avg': round(competitor_avg, 2),
            'difference': round(target_comp_score - competitor_avg, 2),
            'better': target_comp_score > competitor_avg,
        }

    return {
        'overall_rank': target_rank,
        'total_brands': len(all_scores),
        'target_score': target_score,
        'competitor_avg_score': round(sum(c['score'] for c in competitor_scores) / max(len(competitor_scores), 1), 2),
        'score_difference': round(target_score - sum(c['score'] for c in competitor_scores) / max(len(competitor_scores), 1), 2),
        'component_comparison': component_comparison,
        'strengths': strengths[:5],  # Top 5 strengths
        'weaknesses': weaknesses[:5],  # Top 5 weaknesses
        'ranking': all_scores,
    }


def identify_top_countries(
    industry: str,
    client: OpenAI,
    num_countries: int = 5,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, str]]:
    """Identify top countries where the industry has most sales/presence."""

    prompt = f"""
You are a market research analyst helping identify key geographic markets.

INDUSTRY: {industry}

TASK:
Identify the top {num_countries} countries where the {industry} industry has the highest sales, market size, or presence.

For each country, provide:
1. Country name
2. Market size or sales estimate (approximate, with reasoning)
3. Why this country is important for this industry

Return STRICT JSON only, as an array of country objects:

[
  {{
    "country": "Country Name",
    "market_info": "Market size/sales estimate with reasoning...",
    "importance": "Why this country matters for this industry..."
  }},
  ...
]
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = resp.choices[0].message.content
    cleaned = _extract_json_from_text(raw_text)

    try:
        countries = json.loads(cleaned)
        return countries if isinstance(countries, list) else []
    except json.JSONDecodeError:
        print("Failed to parse countries JSON")
        print("RAW OUTPUT:", raw_text[:500])
        return []


def calculate_geographic_presence(
    brand: str,
    industry: str,
    brand_description: str,
    brand_aliases: Optional[List[str]] = None,
    api_key: str = None,
    num_countries: int = 5,
    questions_per_country: int = 5,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Calculate presence scores for the brand across top countries."""

    client = OpenAI(api_key=api_key, timeout=240.0)  # 4 minute timeout for API calls

    # Step 1: Identify top countries for the industry
    top_countries = identify_top_countries(
        industry=industry,
        client=client,
        num_countries=num_countries,
    )

    country_results = []

    # Step 2: For each country, generate country-specific questions and calculate presence
    for country_info in top_countries:
        country = country_info['country']

        # Generate country-specific questions
        country_prompt = f"""
You are helping create questions to test brand visibility in a specific geographic market.

BRAND: {brand}
INDUSTRY: {industry}
COUNTRY: {country}

TASK:
Generate {questions_per_country} questions that a customer in {country} might ask when researching {industry} products/services.

These should be questions that:
- Are relevant to the {country} market
- Might mention local preferences, regulations, or needs
- Could include comparisons with local/regional brands
- Mix general industry questions with country-specific concerns

Return STRICT JSON only, as an array of question strings:

[
  "What are the best {industry} products available in {country}?",
  "Which {industry} brands are popular in {country}?",
  ...
]
"""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": country_prompt}],
        )

        raw_text = resp.choices[0].message.content
        cleaned = _extract_json_from_text(raw_text)

        try:
            questions = json.loads(cleaned)
            if not isinstance(questions, list):
                questions = []
        except json.JSONDecodeError:
            questions = []

        # Get answers to these questions
        country_responses = []
        for q in questions:
            answer_resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q}],
            )
            country_responses.append({
                "question": q,
                "answer": answer_resp.choices[0].message.content,
            })

        # Calculate presence score for this country
        mentions = 0
        total_questions = len(country_responses)

        for resp in country_responses:
            if _brand_mentioned(resp["answer"], brand, brand_aliases):
                mentions += 1

        presence_score = (mentions / max(total_questions, 1)) * 100 if total_questions > 0 else 0

        country_results.append({
            'country': country,
            'market_info': country_info['market_info'],
            'importance': country_info['importance'],
            'presence_score': round(presence_score, 2),
            'mentions': mentions,
            'total_questions': total_questions,
            'questions': questions,
            'responses': country_responses,
        })

    # Calculate overall geographic presence metrics
    avg_presence = sum(c['presence_score'] for c in country_results) / max(len(country_results), 1)
    strong_markets = [c for c in country_results if c['presence_score'] >= 60]
    weak_markets = [c for c in country_results if c['presence_score'] < 40]

    return {
        'brand': brand,
        'industry': industry,
        'num_countries_analyzed': len(country_results),
        'average_presence_score': round(avg_presence, 2),
        'country_results': country_results,
        'strong_markets': strong_markets,
        'weak_markets': weak_markets,
    }


# PDF Generation Functions
def generate_comprehensive_pdf(
    brand: str,
    industry: str,
    visibility_result: Optional[Dict[str, Any]] = None,
    competitive_result: Optional[Dict[str, Any]] = None,
    ranking_result: Optional[Dict[str, Any]] = None,
    geographic_result: Optional[Dict[str, Any]] = None,
) -> io.BytesIO:
    """Generate a comprehensive PDF report with all available analysis results."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=12,
    )
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=10,
        spaceBefore=10,
    )
    normal_style = styles['Normal']

    # Title Page
    story.append(Paragraph(f"AI Visibility Analysis Report", title_style))
    story.append(Paragraph(f"<b>Brand:</b> {brand}", normal_style))
    story.append(Paragraph(f"<b>Industry:</b> {industry}", normal_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    story.append(Spacer(1, 0.3*inch))

    # 1. Visibility Report
    if visibility_result:
        story.append(PageBreak())
        story.append(Paragraph("1. Overall Visibility Score", heading1_style))
        story.append(Spacer(1, 0.1*inch))

        vis = visibility_result['visibility']
        story.append(Paragraph(f"<b>Overall Score:</b> {vis['visibility_score']} / 100", normal_style))
        story.append(Spacer(1, 0.1*inch))

        # Component Scores Table
        story.append(Paragraph("Component Scores", heading2_style))
        comp_data = [['Component', 'Score']]
        for component in ['presence', 'prominence', 'narrative', 'authority']:
            comp_data.append([component.capitalize(), str(vis['component_scores'][component])])

        comp_table = Table(comp_data, colWidths=[3*inch, 2*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 0.2*inch))

        # Brand Config Details
        if 'config' in visibility_result:
            config = visibility_result['config']
            story.append(Paragraph("Auto-Generated Brand Configuration", heading2_style))

            # Aliases
            if hasattr(config, 'aliases') and config.aliases:
                story.append(Paragraph(f"<b>Brand Aliases:</b> {', '.join(config.aliases)}", normal_style))

            # Official URLs
            if hasattr(config, 'official_urls_hint') and config.official_urls_hint:
                story.append(Paragraph(f"<b>Official URLs:</b> {', '.join(config.official_urls_hint)}", normal_style))
            story.append(Spacer(1, 0.1*inch))

            # Narrative Rules
            if hasattr(config, 'narrative_required') and config.narrative_required:
                story.append(Paragraph("<b>Required Narrative Phrases:</b>", normal_style))
                for phrase in config.narrative_required[:10]:
                    story.append(Paragraph(f"  • {phrase}", normal_style))
            story.append(Spacer(1, 0.2*inch))

        # Detailed Question-by-Question Analysis
        if 'raw_responses' in visibility_result:
            story.append(PageBreak())
            story.append(Paragraph("Detailed Question Analysis", heading2_style))
            mentioned_count = sum(1 for d in vis['details'] if d['presence'] > 0)
            total_questions = len(vis['details'])
            story.append(Paragraph(f"Brand mentioned in {mentioned_count} out of {total_questions} questions ({round(mentioned_count/max(total_questions,1)*100)}%)", normal_style))
            story.append(Spacer(1, 0.2*inch))

            for idx, (response, detail) in enumerate(zip(visibility_result['raw_responses'], vis['details']), 1):
                story.append(Paragraph(f"<b>Question {idx}:</b> {response['question']}", normal_style))
                story.append(Spacer(1, 0.05*inch))

                # Scores for this question
                score_text = f"Presence: {detail['presence']*100:.0f}% | Prominence: {detail['prominence']*100:.0f}% | Narrative: {detail['narrative']*100:.0f}% | Authority: {detail['authority']*100:.0f}%"
                story.append(Paragraph(f"<i>{score_text}</i>", normal_style))
                story.append(Spacer(1, 0.05*inch))

                # AI's Answer
                answer_text = response['answer'].replace('<', '&lt;').replace('>', '&gt;')[:1000]
                if len(response['answer']) > 1000:
                    answer_text += "... (truncated)"
                story.append(Paragraph(f"<b>AI Response:</b> {answer_text}", normal_style))
                story.append(Spacer(1, 0.15*inch))

    # 2. Competitive Analysis
    if competitive_result:
        story.append(PageBreak())
        story.append(Paragraph("2. Competitive Analysis", heading1_style))
        story.append(Spacer(1, 0.1*inch))

        analysis = competitive_result['analysis']

        # Overall Ranking
        story.append(Paragraph(f"<b>Your Rank:</b> #{analysis['overall_rank']} of {analysis['total_brands']}", normal_style))
        story.append(Paragraph(f"<b>Your Score:</b> {analysis['target_score']}", normal_style))
        story.append(Paragraph(f"<b>Competitor Average:</b> {analysis['competitor_avg_score']}", normal_style))
        story.append(Paragraph(f"<b>Difference:</b> {'+' if analysis['score_difference'] > 0 else ''}{analysis['score_difference']}", normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Rankings Table
        story.append(Paragraph("Brand Rankings", heading2_style))
        rank_data = [['Rank', 'Brand', 'Score']]
        for idx, item in enumerate(analysis['ranking'], 1):
            brand_name = f"{item['name']} (You)" if item['is_target'] else item['name']
            rank_data.append([f"#{idx}", brand_name, str(item['score'])])

        rank_table = Table(rank_data, colWidths=[1*inch, 3*inch, 1.5*inch])
        rank_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d2ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(rank_table)
        story.append(Spacer(1, 0.2*inch))

        # Component Comparison Details
        story.append(Paragraph("Component-by-Component Comparison", heading2_style))
        comp_comp_data = [['Component', 'Your Score', 'Competitor Avg', 'Difference']]
        for component in ['presence', 'prominence', 'narrative', 'authority']:
            comp_info = analysis['component_comparison'][component]
            diff_symbol = '↑' if comp_info['better'] else '↓'
            comp_comp_data.append([
                component.capitalize(),
                str(comp_info['target']),
                str(comp_info['competitor_avg']),
                f"{diff_symbol} {comp_info['difference']:+.2f}"
            ])

        comp_comp_table = Table(comp_comp_data, colWidths=[1.5*inch, 1.3*inch, 1.5*inch, 1.2*inch])
        comp_comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d2ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(comp_comp_table)
        story.append(Spacer(1, 0.2*inch))

        # Detailed Competitors Information
        if 'competitors' in competitive_result:
            story.append(PageBreak())
            story.append(Paragraph("Competitor Details", heading2_style))

            for idx, comp in enumerate(competitive_result['competitors'], 1):
                story.append(Paragraph(f"<b>Competitor {idx}: {comp['name']}</b>", normal_style))
                story.append(Paragraph(f"<i>Description:</i> {comp['description']}", normal_style))
                story.append(Paragraph(f"<i>Why Competitor:</i> {comp['competitive_reason']}", normal_style))
                story.append(Spacer(1, 0.05*inch))

                # Competitor scores
                comp_vis = comp['visibility']
                story.append(Paragraph(f"Overall Score: {comp_vis['visibility_score']}/100", normal_style))
                comp_scores = f"Presence: {comp_vis['component_scores']['presence']} | Prominence: {comp_vis['component_scores']['prominence']} | Narrative: {comp_vis['component_scores']['narrative']} | Authority: {comp_vis['component_scores']['authority']}"
                story.append(Paragraph(comp_scores, normal_style))
                story.append(Spacer(1, 0.15*inch))

        # Strengths
        if analysis.get('strengths'):
            story.append(PageBreak())
            story.append(Paragraph("Your Strengths (Areas Where You Excel)", heading2_style))
            for idx, strength in enumerate(analysis['strengths'], 1):
                story.append(Paragraph(f"<b>{idx}. {strength['question']}</b>", normal_style))
                score_text = f"Score: {strength['score']}/100 | Presence: {strength['presence']*100:.0f}% | Prominence: {strength['prominence']*100:.0f}% | Narrative: {strength['narrative']*100:.0f}%"
                story.append(Paragraph(f"<i>{score_text}</i>", normal_style))
                story.append(Spacer(1, 0.1*inch))

        # Weaknesses
        if analysis.get('weaknesses'):
            story.append(Paragraph("Areas for Improvement (Weaknesses)", heading2_style))
            for idx, weakness in enumerate(analysis['weaknesses'], 1):
                story.append(Paragraph(f"<b>{idx}. {weakness['question']}</b>", normal_style))
                score_text = f"Score: {weakness['score']}/100 | Presence: {weakness['presence']*100:.0f}% | Prominence: {weakness['prominence']*100:.0f}% | Narrative: {weakness['narrative']*100:.0f}%"
                story.append(Paragraph(f"<i>{score_text}</i>", normal_style))
                story.append(Spacer(1, 0.1*inch))

    # 3. Ranking Analysis
    if ranking_result:
        story.append(PageBreak())
        story.append(Paragraph("3. Ranking Analysis", heading1_style))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(f"<b>Total Queries:</b> {ranking_result['total_prompts']}", normal_style))
        story.append(Paragraph(f"<b>Mention Rate:</b> {ranking_result['mention_rate']}% ({ranking_result['mentioned_count']}/{ranking_result['total_prompts']})", normal_style))
        story.append(Paragraph(f"<b>Average Position:</b> #{ranking_result['average_position']}" if ranking_result['average_position'] else "<b>Average Position:</b> N/A", normal_style))
        story.append(Paragraph(f"<b>Top 3 Appearances:</b> {ranking_result['top_3_count']} ({round(ranking_result['top_3_count']/max(ranking_result['total_prompts'],1)*100)}%)", normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Detailed Ranking Query Results
        if ranking_result.get('all_results'):
            story.append(PageBreak())
            story.append(Paragraph("Detailed Ranking Query Results", heading2_style))

            for idx, item in enumerate(ranking_result['all_results'], 1):
                # Query
                story.append(Paragraph(f"<b>Query {idx}:</b> {item['prompt']}", normal_style))
                story.append(Spacer(1, 0.05*inch))

                # Status
                if item['mentioned']:
                    position_emoji = "🏆" if item.get('in_top_3') else "⭐" if item.get('in_top_5') else "✓"
                    status = f"{position_emoji} Mentioned at position #{item['position']}"
                else:
                    status = "❌ Not mentioned"
                story.append(Paragraph(f"<i>Status: {status}</i>", normal_style))
                story.append(Spacer(1, 0.05*inch))

                # Full AI Response
                if 'answer' in item:
                    answer_text = item['answer'].replace('<', '&lt;').replace('>', '&gt;')[:1500]
                    if len(item['answer']) > 1500:
                        answer_text += "... (truncated)"
                    story.append(Paragraph(f"<b>AI Response:</b> {answer_text}", normal_style))
                story.append(Spacer(1, 0.15*inch))

    # 4. Geographic Analysis
    if geographic_result:
        story.append(PageBreak())
        story.append(Paragraph("4. Geographic Presence Analysis", heading1_style))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(f"<b>Countries Analyzed:</b> {geographic_result['num_countries_analyzed']}", normal_style))
        story.append(Paragraph(f"<b>Average Presence Score:</b> {geographic_result['average_presence_score']}%", normal_style))
        story.append(Paragraph(f"<b>Strong Markets (≥60%):</b> {len(geographic_result['strong_markets'])}", normal_style))
        story.append(Paragraph(f"<b>Weak Markets (<40%):</b> {len(geographic_result['weak_markets'])}", normal_style))
        story.append(Spacer(1, 0.2*inch))

        # Country Results Summary Table
        story.append(Paragraph("Country-by-Country Summary", heading2_style))
        geo_data = [['Country', 'Presence Score', 'Mentions', 'Market Size']]
        for country in geographic_result['country_results']:
            market_info_short = country['market_info'][:50] + "..." if len(country['market_info']) > 50 else country['market_info']
            geo_data.append([
                country['country'],
                f"{country['presence_score']}%",
                f"{country['mentions']}/{country['total_questions']}",
                market_info_short
            ])

        geo_table = Table(geo_data, colWidths=[1.2*inch, 1*inch, 0.8*inch, 2.5*inch])
        geo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#43e97b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(geo_table)
        story.append(Spacer(1, 0.2*inch))

        # Detailed Country Analysis
        if geographic_result.get('country_results'):
            for country_data in geographic_result['country_results']:
                story.append(PageBreak())
                story.append(Paragraph(f"Country Analysis: {country_data['country']}", heading2_style))
                story.append(Paragraph(f"<b>Presence Score:</b> {country_data['presence_score']}% ({country_data['mentions']}/{country_data['total_questions']} mentions)", normal_style))
                story.append(Paragraph(f"<b>Market Information:</b> {country_data['market_info']}", normal_style))
                story.append(Paragraph(f"<b>Industry Importance:</b> {country_data['importance']}", normal_style))
                story.append(Spacer(1, 0.15*inch))

                # Questions and Responses for this country
                if 'responses' in country_data and country_data['responses']:
                    story.append(Paragraph(f"Questions & Responses for {country_data['country']}:", heading2_style))

                    for q_idx, response in enumerate(country_data['responses'], 1):
                        story.append(Paragraph(f"<b>Question {q_idx}:</b> {response['question']}", normal_style))
                        story.append(Spacer(1, 0.05*inch))

                        # AI Response
                        answer_text = response['answer'].replace('<', '&lt;').replace('>', '&gt;')[:1000]
                        if len(response['answer']) > 1000:
                            answer_text += "... (truncated)"
                        story.append(Paragraph(f"<b>AI Response:</b> {answer_text}", normal_style))
                        story.append(Spacer(1, 0.15*inch))

    # Footer
    story.append(PageBreak())
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Generated with AI Visibility Score System", ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


# Flask Routes
@app.route('/health')
def health():
    """Health check endpoint for debugging"""
    import sys
    health_info = {
        'status': 'ok',
        'python_version': sys.version,
        'database_url_exists': bool(os.getenv('DATABASE_URL')),
        'database_url_prefix': os.getenv('DATABASE_URL', 'sqlite:///')[:30] + '...' if os.getenv('DATABASE_URL') else 'sqlite:///ai_visibility.db'
    }
    return jsonify(health_info)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/find-competitors', methods=['POST'])
def find_competitors():
    try:
        data = request.json
        brand = data.get('brand')
        industry = data.get('industry')
        description = data.get('description')
        api_key = data.get('api_key')
        num_competitors = data.get('num_competitors', 5)

        if not all([brand, industry, description, api_key]):
            return jsonify({'error': 'All fields are required'}), 400

        client = OpenAI(api_key=api_key, timeout=240.0)  # 4 minute timeout for API calls

        # Identify competitors
        competitors = identify_competitors(
            brand=brand,
            industry=industry,
            brand_description=description,
            client=client,
            num_competitors=num_competitors,
        )

        return jsonify({'competitors': competitors})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/competitive-analysis', methods=['POST'])
def competitive_analysis():
    try:
        data = request.json
        brand = data.get('brand')
        industry = data.get('industry')
        description = data.get('description')
        api_key = data.get('api_key')
        num_competitors = data.get('num_competitors', 3)

        if not all([brand, industry, description, api_key]):
            return jsonify({'error': 'All fields are required'}), 400

        # Run competitive analysis
        result = run_competitive_analysis(
            brand=brand,
            industry=industry,
            brand_description=description,
            api_key=api_key,
            num_competitors=num_competitors,
        )

        # Save to database (non-blocking - don't fail if DB save fails)
        try:
            save_competitive_analysis(brand, industry, result)
            print(f"✓ Saved competitive analysis for {brand}")
        except Exception as db_error:
            print(f"⚠ Failed to save to database: {db_error}")
            import traceback
            traceback.print_exc()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ranking-analysis', methods=['POST'])
def ranking_analysis():
    try:
        data = request.json
        brand = data.get('brand')
        industry = data.get('industry')
        description = data.get('description')
        api_key = data.get('api_key')
        num_prompts = data.get('num_prompts', 10)

        if not all([brand, industry, description, api_key]):
            return jsonify({'error': 'All fields are required'}), 400

        client = OpenAI(api_key=api_key, timeout=240.0)  # 4 minute timeout for API calls

        # Get brand config to extract aliases
        config = generate_brand_config_via_ai(
            brand=brand,
            industry=industry,
            brand_description=description,
            client=client,
        )

        # Run ranking analysis
        result = analyze_ranking_performance(
            brand=brand,
            industry=industry,
            brand_description=description,
            brand_aliases=config.aliases,
            client=client,
            num_prompts=num_prompts,
        )

        # Save to database (non-blocking - don't fail if DB save fails)
        try:
            save_ranking_analysis(brand, industry, result)
            print(f"✓ Saved ranking analysis for {brand}")
        except Exception as db_error:
            print(f"⚠ Failed to save to database: {db_error}")
            import traceback
            traceback.print_exc()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/geographic-analysis', methods=['POST'])
def geographic_analysis():
    try:
        data = request.json
        brand = data.get('brand')
        industry = data.get('industry')
        description = data.get('description')
        api_key = data.get('api_key')
        num_countries = data.get('num_countries', 5)
        questions_per_country = data.get('questions_per_country', 5)

        if not all([brand, industry, description, api_key]):
            return jsonify({'error': 'All fields are required'}), 400

        client = OpenAI(api_key=api_key, timeout=240.0)  # 4 minute timeout for API calls

        # Get brand config to extract aliases
        config = generate_brand_config_via_ai(
            brand=brand,
            industry=industry,
            brand_description=description,
            client=client,
        )

        # Run geographic presence analysis
        result = calculate_geographic_presence(
            brand=brand,
            industry=industry,
            brand_description=description,
            brand_aliases=config.aliases,
            api_key=api_key,
            num_countries=num_countries,
            questions_per_country=questions_per_country,
        )

        # Save to database (non-blocking - don't fail if DB save fails)
        try:
            save_geographic_score(brand, industry, result)
            print(f"✓ Saved geographic analysis for {brand}")
        except Exception as db_error:
            print(f"⚠ Failed to save to database: {db_error}")
            import traceback
            traceback.print_exc()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        print("=== Generate Report Request ===")
        data = request.json
        brand = data.get('brand')
        industry = data.get('industry')
        description = data.get('description')
        api_key = data.get('api_key')

        print(f"Brand: {brand}, Industry: {industry}")

        if not all([brand, industry, description, api_key]):
            return jsonify({'error': 'All fields are required'}), 400

        # Run the visibility audit
        print("Running visibility audit...")
        result = run_visibility_audit_for_brand(
            brand=brand,
            industry=industry,
            brand_description=description,
            api_key=api_key,
        )
        print("✓ Visibility audit completed")

        # Save to database (non-blocking - don't fail if DB save fails)
        try:
            save_visibility_score(brand, industry, result)
            print(f"✓ Saved visibility score for {brand}")
        except Exception as db_error:
            print(f"⚠ Failed to save to database: {db_error}")
            import traceback
            traceback.print_exc()
            # Continue anyway - don't fail the request

        # Generate markdown report
        print("Generating markdown report...")
        report_md = generate_visibility_report_markdown(result)
        print("✓ Report generated")

        return jsonify({'report': report_md})

    except Exception as e:
        print(f"ERROR in generate_report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        brand = data.get('brand')
        industry = data.get('industry')

        # Get all the analysis results from the request
        visibility_result = data.get('visibility_result')
        competitive_result = data.get('competitive_result')
        ranking_result = data.get('ranking_result')
        geographic_result = data.get('geographic_result')

        if not brand or not industry:
            return jsonify({'error': 'Brand and industry are required'}), 400

        # Generate PDF
        pdf_buffer = generate_comprehensive_pdf(
            brand=brand,
            industry=industry,
            visibility_result=visibility_result,
            competitive_result=competitive_result,
            ranking_result=ranking_result,
            geographic_result=geographic_result,
        )

        # Create filename
        filename = f"AI_Visibility_Report_{brand.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Render the main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/historical-data/<brand_name>', methods=['GET'])
def get_historical_data(brand_name):
    """Get historical data for a brand"""
    try:
        limit = request.args.get('limit', 10, type=int)
        scores = get_historical_visibility_scores(brand_name, limit=limit)

        # Convert to JSON-serializable format
        data = []
        for score in scores:
            data.append({
                'date': score.scan_date.isoformat(),
                'overall_score': score.overall_score,
                'presence_score': score.presence_score,
                'prominence_score': score.prominence_score,
                'narrative_score': score.narrative_score,
                'authority_score': score.authority_score,
                'mentions_count': score.mentions_count,
                'total_questions': score.total_questions,
            })

        return jsonify({'success': True, 'data': data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/latest-scores/<brand_name>', methods=['GET'])
def get_latest_scores_api(brand_name):
    """Get latest scores across all analysis types"""
    try:
        scores = get_latest_scores(brand_name)

        result = {}

        # Visibility
        if scores['visibility']:
            v = scores['visibility']
            result['visibility'] = {
                'date': v.scan_date.isoformat(),
                'overall_score': v.overall_score,
                'component_scores': {
                    'presence': v.presence_score,
                    'prominence': v.prominence_score,
                    'narrative': v.narrative_score,
                    'authority': v.authority_score,
                },
                'mentions_count': v.mentions_count,
                'total_questions': v.total_questions,
            }

        # Competitive
        if scores['competitive']:
            c = scores['competitive']
            result['competitive'] = {
                'date': c.scan_date.isoformat(),
                'rank': c.overall_rank,
                'total_brands': c.total_brands,
                'score': c.target_score,
                'competitor_avg': c.competitor_avg_score,
                'difference': c.score_difference,
            }

        # Ranking
        if scores['ranking']:
            r = scores['ranking']
            result['ranking'] = {
                'date': r.scan_date.isoformat(),
                'total_prompts': r.total_prompts,
                'mention_rate': r.mention_rate,
                'average_position': r.average_position,
                'top_3_count': r.top_3_count,
            }

        # Geographic
        if scores['geographic']:
            g = scores['geographic']
            result['geographic'] = {
                'date': g.scan_date.isoformat(),
                'num_countries': g.num_countries_analyzed,
                'avg_presence': g.average_presence_score,
                'strong_markets': g.strong_markets_count,
                'weak_markets': g.weak_markets_count,
            }

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/brands', methods=['GET'])
def get_brands_api():
    """Get all brands that have been analyzed"""
    try:
        brands = get_all_brands()
        return jsonify({'success': True, 'brands': brands})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='127.0.0.1')
