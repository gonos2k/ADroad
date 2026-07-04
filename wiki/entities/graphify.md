---
title: "graphify (그래프 추출 백엔드)"
instance_of: Artifact
page_kind: entity-page
epistemic_status: observed
---

# graphify (그래프 추출 백엔드)

## Role

tree-sitter 기반 구조 그래프 추출기(graphifyy). kg-skill의 그래프 레이어 백엔드로, Fortran 포함 25+ 언어의 MODULE/SUBROUTINE/USE/CALL 그래프를 로컬에서 추출.

## Key Facts

- 설치: uv tool install graphifyy (권장) 또는 pip install 'graphifyy>=0.8.24'
- reverse-impact 'affected'(blast radius), query --context 엣지 필터, MCP over stdio
- 위키 정리·BM25 검색은 graphify 없이도 동작; 그래프 순회·merge·mcp만 의존

## Connections

kg 위키의 별도 레이어. dROAD 코드 자체와는 독립. 관련 위키: [[droad]].
