# dROAD 기술보고서 — 빌드 및 provenance

`dROAD_report.md`가 **source of truth**다. DOCX/PDF는 여기서 생성되는 산출물이며, 내용 리뷰·diff·재현은 Markdown 기준으로 한다.

## 파일
| 파일 | 역할 |
|---|---|
| `dROAD_report.md` | 보고서 원본(Markdown, 리뷰·diff 대상) |
| `figures/make_figures.py` | 그림 4종 생성 스크립트(matplotlib) |
| `figures/*.png` | 생성된 그림 원본(아키텍처·DA 사이클·단일/다중 window) |
| `build_docx.js` | DOCX 생성기(docx-js) — Markdown 내용과 동기 유지 |
| `../../dROAD_기술보고서.docx` / `.pdf` | 배포 산출물(repo 루트) |

## 재생성
```bash
# 그림
python docs/report/figures/make_figures.py      # figures/*.png

# DOCX (Node + docx 필요: npm install -g docx)
node docs/report/build_docx.js                  # dROAD_기술보고서.docx

# PDF (LibreOffice)
soffice --headless --convert-to pdf dROAD_기술보고서.docx
```

## Provenance
- **Version**: v1.0 · **Date**: 2026-07-06
- **Content basis (commit)**: `a9f635f`
- 실험 수치 출처: `reports/*.md`, `reports/*_meta.json` (코어 187 + jax 34 통과, 감사 clean)

## 유지보수 원칙
- 실험 결과·수식·문구 변경은 **`dROAD_report.md`에서 먼저** 하고, 이후 `build_docx.js`에 반영해 DOCX/PDF를 재생성한다.
- 보고서 결론은 코드의 보수적 판정과 일치해야 한다: forecast DA는 **report-only milestone**(단일 window 개선, multi-window 2/4 재현, promotion 불가).
