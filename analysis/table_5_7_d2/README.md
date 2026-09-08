# 표 5-7·부록 D-2 다시 계산

보존된 기업명 인식 응답과 IC-b·IC-c Stage 8 결과를 결합해 36개 셀을 다시 계산.

입력 위치·분석 조건·순열 횟수·난수 시작값: `analysis_settings.json`.

NumPy, pandas, PyArrow가 설치된 Python 환경에서 실행:

```powershell
python run_independent_analysis.py `
  --source-root C:\path\to\scientific-source-root `
  --output-dir C:\new\output-directory
```

생성 파일: 결합 입력 CSV, 36셀 결과 CSV, 요약 JSON, 실행 기록 JSON.
