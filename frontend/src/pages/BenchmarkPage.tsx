import React, { useState } from 'react';
import Layout from '../components/Layout';
import api from '../services/api';

const BenchmarkPage: React.FC = () => {
  const [questions, setQuestions] = useState<string[]>(['']);
  const [answers, setAnswers] = useState<string[]>(['']);
  const [contexts, setContexts] = useState<string[][]>(['']);
  const [groundTruths, setGroundTruths] = useState<string[]>(['']);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const addQuestion = () => {
    setQuestions([...questions, '']);
    setAnswers([...answers, '']);
    setContexts([...contexts, ['']]);
    setGroundTruths([...groundTruths, '']);
  };

  const removeQuestion = (index: number) => {
    setQuestions(questions.filter((_, i) => i !== index));
    setAnswers(answers.filter((_, i) => i !== index));
    setContexts(contexts.filter((_, i) => i !== index));
    setGroundTruths(groundTruths.filter((_, i) => i !== index));
  };

  const handleQuestionChange = (index: number, value: string) => {
    const newQuestions = [...questions];
    newQuestions[index] = value;
    setQuestions(newQuestions);
  };

  const handleAnswerChange = (index: number, value: string) => {
    const newAnswers = [...answers];
    newAnswers[index] = value;
    setAnswers(newAnswers);
  };

  const handleContextChange = (index: number, contextIndex: number, value: string) => {
    const newContexts = [...contexts];
    newContexts[index][contextIndex] = value;
    setContexts(newContexts);
  };

  const addContext = (index: number) => {
    const newContexts = [...contexts];
    newContexts[index] = [...newContexts[index], ''];
    setContexts(newContexts);
  };

  const handleGroundTruthChange = (index: number, value: string) => {
    const newGroundTruths = [...groundTruths];
    newGroundTruths[index] = value;
    setGroundTruths(newGroundTruths);
  };

  const handleRunBenchmark = async () => {
    // Filter out empty entries
    const filteredQuestions = questions.filter((q) => q.trim());
    const filteredAnswers = answers.filter((a) => a.trim());
    const filteredContexts = contexts
      .map((ctx) => ctx.filter((c) => c.trim()))
      .filter((ctx) => ctx.length > 0);
    const filteredGroundTruths = groundTruths.filter((gt) => gt.trim());

    if (
      filteredQuestions.length === 0 ||
      filteredQuestions.length !== filteredAnswers.length ||
      filteredQuestions.length !== filteredContexts.length
    ) {
      alert('Bitte füllen Sie alle Felder aus');
      return;
    }

    setLoading(true);
    try {
      const response = await api.post('/api/benchmark/run', {
        questions: filteredQuestions,
        answers: filteredAnswers,
        contexts: filteredContexts,
        ground_truths: filteredGroundTruths.length > 0 ? filteredGroundTruths : undefined,
      });
      setResults(response.data);
    } catch (error) {
      console.error('Error running benchmark:', error);
      alert('Fehler beim Ausführen des Benchmarks');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div className="px-4 py-6 sm:px-0">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">RAGAS Benchmarking</h2>

        <div className="space-y-6">
          {questions.map((question, index) => (
            <div key={index} className="bg-white shadow rounded-lg p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">Frage {index + 1}</h3>
                {questions.length > 1 && (
                  <button
                    onClick={() => removeQuestion(index)}
                    className="text-red-600 hover:text-red-800"
                  >
                    Entfernen
                  </button>
                )}
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Frage
                  </label>
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => handleQuestionChange(index, e.target.value)}
                    className="w-full rounded-md border-gray-300 shadow-sm"
                    placeholder="Geben Sie eine Frage ein..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Antwort
                  </label>
                  <textarea
                    value={answers[index]}
                    onChange={(e) => handleAnswerChange(index, e.target.value)}
                    className="w-full rounded-md border-gray-300 shadow-sm"
                    rows={3}
                    placeholder="Geben Sie die Antwort ein..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Kontexte
                  </label>
                  {contexts[index].map((context, ctxIndex) => (
                    <div key={ctxIndex} className="mb-2">
                      <textarea
                        value={context}
                        onChange={(e) =>
                          handleContextChange(index, ctxIndex, e.target.value)
                        }
                        className="w-full rounded-md border-gray-300 shadow-sm"
                        rows={2}
                        placeholder={`Kontext ${ctxIndex + 1}...`}
                      />
                    </div>
                  ))}
                  <button
                    onClick={() => addContext(index)}
                    className="mt-2 text-sm text-indigo-600 hover:text-indigo-800"
                  >
                    + Kontext hinzufügen
                  </button>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Ground Truth (optional)
                  </label>
                  <textarea
                    value={groundTruths[index]}
                    onChange={(e) => handleGroundTruthChange(index, e.target.value)}
                    className="w-full rounded-md border-gray-300 shadow-sm"
                    rows={2}
                    placeholder="Geben Sie die erwartete Antwort ein..."
                  />
                </div>
              </div>
            </div>
          ))}

          <div className="flex space-x-4">
            <button
              onClick={addQuestion}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
            >
              + Frage hinzufügen
            </button>
            <button
              onClick={handleRunBenchmark}
              disabled={loading}
              className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50"
            >
              {loading ? 'Läuft...' : 'Benchmark ausführen'}
            </button>
          </div>

          {results && (
            <div className="bg-white shadow rounded-lg p-6 mt-6">
              <h3 className="text-lg font-medium mb-4">Ergebnisse</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Faithfulness</p>
                  <p className="text-2xl font-bold">
                    {(results.summary.faithfulness * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Answer Relevancy</p>
                  <p className="text-2xl font-bold">
                    {(results.summary.answer_relevancy * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Context Precision</p>
                  <p className="text-2xl font-bold">
                    {(results.summary.context_precision * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Context Recall</p>
                  <p className="text-2xl font-bold">
                    {(results.summary.context_recall * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              {results.plot_path && (
                <div className="mt-4">
                  <a
                    href={results.plot_path}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-600 hover:text-indigo-800"
                  >
                    Vollständiges Dashboard anzeigen
                  </a>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default BenchmarkPage;


