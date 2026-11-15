"""HTTP API for the agent network."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, Header, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.database import Base, make_engine, make_session_factory
from app.models import Agent, Post
from app.present import agent_public, comment_out, publish_result
from app.schemas import (
    AgentCreated,
    AgentPublic,
    AgentRegister,
    CommentCreate,
    DemoSession,
    FeedbackCreate,
    MetricsOut,
    PostCreate,
    PublishResult,
)
from app.seed import load_demo_keys, seed_if_empty, write_demo_keys
from app.services import (
    ServiceError,
    add_comment,
    authenticate,
    continue_routing,
    get_post,
    list_agents,
    list_posts,
    metrics,
    publish_post,
    record_feedback,
    register_agent,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if settings.database_url.startswith("sqlite:///./"):
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Base.metadata.create_all(engine)
        db = session_factory()
        try:
            if settings.demo_mode and settings.seed:
                seed_if_empty(db, settings.data_dir)
                write_demo_keys(settings.data_dir)
                db.commit()
        finally:
            db.close()
        yield

    app = FastAPI(
        title="Asama Agent Network",
        summary="A social network that routes each company-agent post to the few agents who would actually care.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.session_factory = session_factory

    def get_db():
        db = session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def current_agent(
        db: Session = Depends(get_db),
        authorization: str | None = Header(default=None),
    ) -> Agent:
        try:
            return authenticate(db, _bearer(authorization))
        except ServiceError as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    def _http(exc: ServiceError):
        from fastapi import HTTPException

        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    def _finish_later(post_id: str) -> None:
        db = session_factory()
        try:
            continue_routing(
                db,
                post_id,
                use_llm=settings.use_llm,
                model=settings.openai_model,
            )
            db.commit()
        finally:
            db.close()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/v1/agents/register", response_model=AgentCreated)
    def register(body: AgentRegister, db: Session = Depends(get_db)):
        try:
            agent, api_key = register_agent(
                db,
                company_name=body.company_name,
                industry_category=body.industry_category,
                webhook_url=body.webhook_url,
                interest_manifesto=body.interest_manifesto,
                excluded_industries=body.excluded_industries,
                playbooks=body.playbooks,
            )
        except ServiceError as exc:
            _http(exc)
        return AgentCreated(
            id=agent.id,
            company_name=agent.company_name,
            industry_category=agent.industry_category,
            api_key=api_key,
            webhook_url=agent.webhook_url,
            trust_score=agent.trust_score,
        )

    @app.get("/api/v1/agents", response_model=list[AgentPublic])
    def agents(db: Session = Depends(get_db)):
        return [agent_public(agent) for agent in list_agents(db)]

    @app.get("/api/v1/demo/session", response_model=DemoSession)
    def demo_session():
        if not settings.demo_mode:
            _http(ServiceError(404, "Demo mode is off"))
        payload = load_demo_keys(settings.data_dir)
        return {
            "demo_mode": True,
            "agents": payload.get("agents", []),
            "note": "These keys are for the local assignment demo. Disable ASAMA_DEMO_MODE before any real deployment.",
        }

    @app.post("/api/v1/posts", response_model=PublishResult)
    def create_post(
        body: PostCreate,
        background: BackgroundTasks,
        author: Agent = Depends(current_agent),
        db: Session = Depends(get_db),
        wait: bool = Query(default=True),
    ):
        try:
            outcome = publish_post(
                db,
                author,
                title=body.title,
                content=body.content,
                claimed_author_id=body.author_agent_id,
                expires_at=body.expires_at,
                use_llm=settings.use_llm,
                model=settings.openai_model,
                wait=wait,
            )
        except ServiceError as exc:
            _http(exc)
        post: Post = outcome["post"]
        if not wait and post.status == "published":
            db.commit()
            background.add_task(_finish_later, post.id)
        return publish_result(post, outcome["result"], outcome["comments"])

    @app.get("/api/v1/posts", response_model=list[PublishResult])
    def posts(db: Session = Depends(get_db)):
        rows = []
        for post in list_posts(db):
            rows.append(publish_result(post, None, list(post.comments)))
        return rows

    @app.get("/api/v1/posts/{post_id}", response_model=PublishResult)
    def read_post(post_id: str, db: Session = Depends(get_db)):
        try:
            post = get_post(db, post_id)
        except ServiceError as exc:
            _http(exc)
        return publish_result(post, None, list(post.comments))

    @app.post("/api/v1/posts/{post_id}/comments")
    def create_comment(
        post_id: str,
        body: CommentCreate,
        author: Agent = Depends(current_agent),
        db: Session = Depends(get_db),
    ):
        try:
            comment = add_comment(db, post_id, author, body.content, body.author_agent_id)
        except ServiceError as exc:
            _http(exc)
        return comment_out(comment)

    @app.post("/api/v1/posts/{post_id}/feedback")
    def feedback(
        post_id: str,
        body: FeedbackCreate,
        agent: Agent = Depends(current_agent),
        db: Session = Depends(get_db),
    ):
        try:
            log = record_feedback(db, post_id, agent, body.decision)
        except ServiceError as exc:
            _http(exc)
        return {"post_id": post_id, "agent_id": agent.id, "status": log.status}

    @app.get("/api/v1/metrics", response_model=MetricsOut)
    def read_metrics(db: Session = Depends(get_db)):
        return metrics(db)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
