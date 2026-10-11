from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import ChatMessage, Conversation, User
from ..schemas import ChatMessageOut, ConversationOut, ConversationRename
from .deps import get_current_user

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _owned(db: Session, conv_id: int, user: User) -> Conversation:
    c = db.scalar(select(Conversation).where(Conversation.id == conv_id,
                                             Conversation.user_id == user.id))
    if not c:
        raise HTTPException(404, "Conversation not found")
    return c


@router.get("", response_model=list[ConversationOut])
def list_conversations(limit: int = Query(30, ge=1, le=200), db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    return db.scalars(select(Conversation).where(Conversation.user_id == user.id)
                      .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                      .limit(limit)).all()


@router.get("/{conv_id}/messages", response_model=list[ChatMessageOut])
def messages(conv_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _owned(db, conv_id, user)
    return db.scalars(select(ChatMessage).where(ChatMessage.conversation_id == conv_id,
                                                ChatMessage.user_id == user.id)
                      .order_by(ChatMessage.id)).all()


@router.patch("/{conv_id}", response_model=ConversationOut)
def rename(conv_id: int, body: ConversationRename, db: Session = Depends(get_db),
           user: User = Depends(get_current_user)):
    c = _owned(db, conv_id, user)
    c.title = body.title.strip()
    c.updated_at = datetime.now(timezone.utc)
    db.commit()
    return c


@router.delete("/{conv_id}", status_code=204)
def delete(conv_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db.delete(_owned(db, conv_id, user))   # messages go via ON DELETE CASCADE
    db.commit()
