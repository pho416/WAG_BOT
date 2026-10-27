import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from wattbot.evaluation import evaluate, validate
from wattbot.corpus import ingest
from wattbot.generation import checked_answer
from wattbot.io import questions, read_jsonl, write_csv
from wattbot.reasoning import decimal_calculate
from wattbot.retrieval import BM25, rrf, select_context


class PipelineTests(unittest.TestCase):
    def test_gold_never_enters_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            write_csv(path, [{"id": "q1", "question": "How much?", "answer_unit": "MWh",
                              "answer_value": "SECRET", "ref_id": "SECRET", "supporting_materials": "SECRET"}])
            self.assertEqual(set(questions(path)[0]), {"id", "question", "answer_unit", "Cohort"})
            self.assertNotIn("SECRET", json.dumps(questions(path)))

    def test_precise_numeric_and_no_code_execution(self):
        result = decimal_calculate("training / (total / households)",
                                   {"training": "1287", "total": "377685", "households": "35000"})
        self.assertLess(abs(Decimal(result) - Decimal("119.26605504587156")), Decimal("0.000000001"))
        self.assertEqual(decimal_calculate("0.1 + 0.2", {}), "0.3")
        for expression in ("__import__('os').getcwd()", "x.__class__", "[x for x in []]", "2**999999", "1/0"):
            with self.assertRaises((ValueError, ArithmeticError)):
                decimal_calculate(expression, {"x": "1"})

    def test_source_quotes_citations_and_calculator(self):
        question = {"id": "q1", "question": "Energy per household?", "answer_unit": "MWh"}
        context = [{"chunk_id": "paper:p1:w0", "ref_id": "paper", "page": 1,
                    "text": "The total energy was 377,685 MWh for 35,000 households."}]
        quote = context[0]["text"]
        source = lambda value: {"value": value, "evidence_id": "paper:p1:w0", "quote": quote}
        result = {"answer_value": "999", "explanation": "Divide total by households.",
                  "citations": [{"evidence_id": "paper:p1:w0", "quote": quote}],
                  "calculation": {"expression": "energy/households", "operands": {
                      "energy": source("377685"), "households": source("35000")}}}
        checked = checked_answer(question, result, context, {"paper": {"url": "https://example.org/paper.pdf"}})
        self.assertTrue(checked["answer_value"].startswith("10.791"))
        self.assertEqual(json.loads(checked["ref_id"]), ["paper"])
        result["calculation"]["operands"]["energy"]["value"] = "123456"
        with self.assertRaises(ValueError):
            checked_answer(question, result, context, {"paper": {"url": "test"}})
        result["citations"][0]["evidence_id"] = "invented"
        with self.assertRaises(ValueError):
            checked_answer(question, result, context, {})

    def test_categorical_unit_is_not_abstention(self):
        q = {"id": "q1", "answer_unit": "is_blank"}
        context = [{"chunk_id": "p:1", "ref_id": "p", "page": 1, "text": "The company is Alphabet."}]
        result = {"answer_value": "Alphabet", "explanation": "Company named in source.",
                  "citations": [{"evidence_id": "p:1", "quote": "The company is Alphabet."}]}
        row = checked_answer(q, result, context, {"p": {"url": "https://example.org"}})
        self.assertEqual(row["answer_value"], "Alphabet")

    def test_retrieval_and_rrf(self):
        bm25 = BM25(["Training consumed electricity", "Water consumption in Arizona"])
        self.assertGreater(bm25.scores("electricity training")[0], bm25.scores("electricity training")[1])
        self.assertEqual(bm25.scores("nonexistent").sum(), 0)
        self.assertGreater(rrf([[0, 1], [1, 2]])[1], rrf([[0, 1], [1, 2]])[0])
        selected = select_context([{"ref_id": "p", "page": 1}, {"ref_id": "p", "page": 1}, {"ref_id": "q", "page": 2}], 2)
        self.assertEqual([x["ref_id"] for x in selected], ["p", "q"])

    def test_strict_validator_rejects_partial_ids_and_hedged_abstention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "q.csv", [{"id": "q1", "question": "Q?", "answer_unit": "MWh"},
                                        {"id": "q2", "question": "Q2?", "answer_unit": "MWh"}])
            write_csv(root / "m.csv", [{"id": "p", "url": "https://example.org/p.pdf"}], fields=("id", "url"))
            row = {"id": "q1", "answer_value": "is_blank", "answer_unit": "MWh", "ref_id": "is_blank",
                   "ref_url": "is_blank", "supporting_materials": "is_blank", "explanation": "No evidence."}
            write_csv(root / "s.csv", [row])
            with self.assertRaises(ValueError):
                validate(root / "s.csv", root / "q.csv", root / "m.csv")
            second = {**row, "id": "q2", "supporting_materials": "An invented quote"}
            write_csv(root / "s.csv", [row, second])
            with self.assertRaises(ValueError):
                validate(root / "s.csv", root / "q.csv", root / "m.csv")

    def test_pdf_ingestion_keeps_page_provenance(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "metadata.csv", [{"id": "p", "title": "Energy study", "year": "2026", "url": "https://example.org/p.pdf"}],
                      fields=("id", "title", "year", "url"))
            with pymupdf.open() as doc:
                doc.new_page().insert_text((50, 50), "Page one energy consumption 100 MWh.")
                doc.new_page().insert_text((50, 50), "Page two water consumption 500 liters.")
                doc.save(root / "p.pdf")
            report = ingest(root / "metadata.csv", root, root / "processed")
            chunks = read_jsonl(root / "processed/chunks.jsonl")
            self.assertEqual(report["pages"], 2)
            self.assertEqual([c["page"] for c in chunks], [1, 2])
            self.assertTrue(all(c["ref_id"] == "p" and c["bboxes"] for c in chunks))
            self.assertNotIn("500", chunks[0]["text"])

    def test_evaluator_uses_2026_weights_for_partial_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"id": "q1", "question": "Amount?", "answer_unit": "MWh", "answer_value": "10",
                   "ref_id": '["p", "q"]', "ref_url": '["https://p", "https://q"]',
                   "supporting_materials": "The reported amount was 10 MWh.", "explanation": "Directly stated.", "Quote": "1"}
            write_csv(root / "gold.csv", [row], fields=list(row))
            write_csv(root / "sub.csv", [{**row, "ref_id": '["p"]', "ref_url": '["https://p"]'}])
            write_csv(root / "metadata.csv", [{"id": "p", "url": "https://p"}, {"id": "q", "url": "https://q"}], fields=("id", "url"))
            scorer = Path(__file__).resolve().parents[1] / "data/raw/Score.py"
            if not scorer.exists():
                self.skipTest("Official scorer requires the user's dataset archive")
            report = evaluate(root / "gold.csv", root / "sub.csv", root / "metadata.csv", scorer, root / "eval")
            expected = 0.75 + 0.20 * (2 / 3) + 0.05
            self.assertAlmostEqual(report["official_score"], expected)
            self.assertAlmostEqual(report["by_type"]["Quote"]["score_2026"], expected)


if __name__ == "__main__":
    unittest.main()
