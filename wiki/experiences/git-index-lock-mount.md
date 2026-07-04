---
title: "마운트 폴더 .git 락으로 커밋이 막힌 사건"
instance_of: Experience
page_kind: experience-page
date: 2026-07-04
---

# 마운트 폴더 .git 락으로 커밋이 막힌 사건

## Context

샌드박스에서 dROAD 워크스페이스(마운트) 안의 git repo에 커밋 시도.

## Attempted

git add/commit; 이후 plumbing(write-tree/commit-tree/update-ref) 우회도 시도.

## Outcome

macOS fuse 마운트가 .git/index.lock·HEAD.lock·refs/heads/main.lock의 unlink를 'Operation not permitted'로 거부. 커밋 객체는 생성되나 ref 이동 실패, 이후 stale 락이 다음 커밋을 차단.

## Root Cause

마운트 계층이 락 파일 삭제를 막아, 샌드박스에서는 git ref 갱신을 완료할 수 없다.

## Resolution

파일 변경은 디스크에 모두 저장됨을 확인시키고, 사용자 로컬에서 `find .git -name '*.lock' -delete` 후 commit·push 하도록 안내. 샌드박스 git 쓰기는 지양.

## Lesson

마운트 repo에서는 파일 산출물까지만 책임지고, git ref 조작은 사용자 로컬에 위임한다.
