"""
Database models for AI Visibility Score tracking
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class Brand(Base):
    """Master brand table"""
    __tablename__ = 'brands'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    industry = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VisibilityScore(Base):
    """Historical visibility scores"""
    __tablename__ = 'visibility_scores'

    id = Column(Integer, primary_key=True)
    brand_name = Column(String(255), nullable=False)
    industry = Column(String(255), nullable=False)
    scan_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Scores
    overall_score = Column(Float)
    presence_score = Column(Float)
    prominence_score = Column(Float)
    narrative_score = Column(Float)
    authority_score = Column(Float)

    # Metadata
    total_questions = Column(Integer)
    mentions_count = Column(Integer)

    # Full result as JSON
    full_result = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)


class CompetitiveAnalysis(Base):
    """Historical competitive analysis"""
    __tablename__ = 'competitive_analysis'

    id = Column(Integer, primary_key=True)
    brand_name = Column(String(255), nullable=False)
    industry = Column(String(255), nullable=False)
    scan_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Rankings
    overall_rank = Column(Integer)
    total_brands = Column(Integer)
    target_score = Column(Float)
    competitor_avg_score = Column(Float)
    score_difference = Column(Float)

    # Full result as JSON
    full_result = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)


class RankingAnalysis(Base):
    """Historical ranking analysis"""
    __tablename__ = 'ranking_analysis'

    id = Column(Integer, primary_key=True)
    brand_name = Column(String(255), nullable=False)
    industry = Column(String(255), nullable=False)
    scan_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Metrics
    total_prompts = Column(Integer)
    mentioned_count = Column(Integer)
    mention_rate = Column(Float)
    average_position = Column(Float)
    top_3_count = Column(Integer)
    top_5_count = Column(Integer)

    # Full result as JSON
    full_result = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)


class GeographicScore(Base):
    """Historical geographic presence scores"""
    __tablename__ = 'geographic_scores'

    id = Column(Integer, primary_key=True)
    brand_name = Column(String(255), nullable=False)
    industry = Column(String(255), nullable=False)
    scan_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Metrics
    num_countries_analyzed = Column(Integer)
    average_presence_score = Column(Float)
    strong_markets_count = Column(Integer)
    weak_markets_count = Column(Integer)

    # Full result as JSON
    full_result = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
def get_database_url():
    """Get database URL from environment or use SQLite for local dev"""
    return os.getenv('DATABASE_URL', 'sqlite:///ai_visibility.db')


def init_db():
    """Initialize database and create tables"""
    try:
        database_url = get_database_url()
        print(f"Database URL (masked): {database_url[:20]}...")

        # Fix for Railway PostgreSQL URL (postgres:// -> postgresql://)
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
            print("✓ Fixed postgres:// to postgresql://")

        engine = create_engine(database_url, pool_pre_ping=True, echo=False)
        Base.metadata.create_all(engine)
        print("✓ Database tables created successfully")
        return engine
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        # Return None instead of crashing - app can still work without DB
        return None


def get_session():
    """Get database session"""
    try:
        database_url = get_database_url()

        # Fix for Railway PostgreSQL URL
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)

        engine = create_engine(database_url, pool_pre_ping=True, echo=False)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        print(f"⚠️ Failed to create database session: {e}")
        raise


def _make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format"""
    import json
    from dataclasses import is_dataclass, asdict

    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif is_dataclass(obj):
        # Convert dataclass to dict
        return _make_json_serializable(asdict(obj))
    else:
        # For other objects, try to convert to string or skip
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def save_visibility_score(brand_name, industry, result):
    """Save visibility score to database"""
    session = get_session()
    try:
        import json
        vis = result['visibility']

        # Convert result to JSON-serializable format (remove non-serializable objects)
        serializable_result = _make_json_serializable({
            'visibility': vis,
            'raw_responses': result.get('raw_responses', [])
        })

        # Test if it's serializable
        json.dumps(serializable_result)

        score = VisibilityScore(
            brand_name=brand_name,
            industry=industry,
            overall_score=vis['visibility_score'],
            presence_score=vis['component_scores']['presence'],
            prominence_score=vis['component_scores']['prominence'],
            narrative_score=vis['component_scores']['narrative'],
            authority_score=vis['component_scores']['authority'],
            total_questions=len(vis['details']),
            mentions_count=sum(1 for d in vis['details'] if d['presence'] > 0),
            full_result=serializable_result
        )
        session.add(score)
        session.commit()
        print(f"✓ Successfully saved visibility score with ID: {score.id}")
        return score.id
    except Exception as e:
        session.rollback()
        print(f"❌ Error saving visibility score: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


def save_competitive_analysis(brand_name, industry, result):
    """Save competitive analysis to database"""
    session = get_session()
    try:
        import json
        analysis = result['analysis']

        # Create JSON-serializable copy (recursively clean all BrandConfig and other objects)
        serializable_result = _make_json_serializable({
            'analysis': analysis,
            'competitors': result.get('competitors', [])
        })

        # Test serialization
        json.dumps(serializable_result)

        comp = CompetitiveAnalysis(
            brand_name=brand_name,
            industry=industry,
            overall_rank=analysis['overall_rank'],
            total_brands=analysis['total_brands'],
            target_score=analysis['target_score'],
            competitor_avg_score=analysis['competitor_avg_score'],
            score_difference=analysis['score_difference'],
            full_result=serializable_result
        )
        session.add(comp)
        session.commit()
        print(f"✓ Successfully saved competitive analysis with ID: {comp.id}")
        return comp.id
    except Exception as e:
        session.rollback()
        print(f"❌ Error saving competitive analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


def save_ranking_analysis(brand_name, industry, result):
    """Save ranking analysis to database"""
    session = get_session()
    try:
        import json

        # Create JSON-serializable copy
        serializable_result = _make_json_serializable({
            'total_prompts': result['total_prompts'],
            'mentioned_count': result['mentioned_count'],
            'mention_rate': result['mention_rate'],
            'average_position': result.get('average_position'),
            'top_3_count': result['top_3_count'],
            'top_5_count': result['top_5_count'],
            'all_results': result.get('all_results', [])
        })

        # Test serialization
        json.dumps(serializable_result)

        ranking = RankingAnalysis(
            brand_name=brand_name,
            industry=industry,
            total_prompts=result['total_prompts'],
            mentioned_count=result['mentioned_count'],
            mention_rate=result['mention_rate'],
            average_position=result.get('average_position'),
            top_3_count=result['top_3_count'],
            top_5_count=result['top_5_count'],
            full_result=serializable_result
        )
        session.add(ranking)
        session.commit()
        print(f"✓ Successfully saved ranking analysis with ID: {ranking.id}")
        return ranking.id
    except Exception as e:
        session.rollback()
        print(f"❌ Error saving ranking analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


def save_geographic_score(brand_name, industry, result):
    """Save geographic score to database"""
    session = get_session()
    try:
        import json

        # Create JSON-serializable copy
        serializable_result = _make_json_serializable({
            'num_countries_analyzed': result['num_countries_analyzed'],
            'average_presence_score': result['average_presence_score'],
            'strong_markets': result['strong_markets'],
            'weak_markets': result['weak_markets'],
            'country_results': result.get('country_results', [])
        })

        # Test serialization
        json.dumps(serializable_result)

        geo = GeographicScore(
            brand_name=brand_name,
            industry=industry,
            num_countries_analyzed=result['num_countries_analyzed'],
            average_presence_score=result['average_presence_score'],
            strong_markets_count=len(result['strong_markets']),
            weak_markets_count=len(result['weak_markets']),
            full_result=serializable_result
        )
        session.add(geo)
        session.commit()
        print(f"✓ Successfully saved geographic score with ID: {geo.id}")
        return geo.id
    except Exception as e:
        session.rollback()
        print(f"❌ Error saving geographic score: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


def get_historical_visibility_scores(brand_name, limit=10):
    """Get historical visibility scores for a brand"""
    session = get_session()
    try:
        scores = session.query(VisibilityScore)\
            .filter_by(brand_name=brand_name)\
            .order_by(VisibilityScore.scan_date.desc())\
            .limit(limit)\
            .all()
        return scores
    finally:
        session.close()


def get_latest_scores(brand_name):
    """Get latest scores across all analysis types"""
    session = get_session()
    try:
        latest_visibility = session.query(VisibilityScore)\
            .filter_by(brand_name=brand_name)\
            .order_by(VisibilityScore.scan_date.desc())\
            .first()

        latest_competitive = session.query(CompetitiveAnalysis)\
            .filter_by(brand_name=brand_name)\
            .order_by(CompetitiveAnalysis.scan_date.desc())\
            .first()

        latest_ranking = session.query(RankingAnalysis)\
            .filter_by(brand_name=brand_name)\
            .order_by(RankingAnalysis.scan_date.desc())\
            .first()

        latest_geographic = session.query(GeographicScore)\
            .filter_by(brand_name=brand_name)\
            .order_by(GeographicScore.scan_date.desc())\
            .first()

        return {
            'visibility': latest_visibility,
            'competitive': latest_competitive,
            'ranking': latest_ranking,
            'geographic': latest_geographic
        }
    finally:
        session.close()


def get_all_brands():
    """Get all unique brands that have been analyzed"""
    session = get_session()
    try:
        from sqlalchemy import distinct

        # Get unique brand names from visibility_scores table
        brands = session.query(
            distinct(VisibilityScore.brand_name),
            VisibilityScore.industry
        ).order_by(VisibilityScore.brand_name).all()

        return [{'name': brand[0], 'industry': brand[1]} for brand in brands]
    finally:
        session.close()
