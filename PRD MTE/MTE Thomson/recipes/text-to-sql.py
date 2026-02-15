# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# text_to_sql.py
# Agent for MTE inventory and procurement analysis via DuckDB (df_main table)
# Features: SQL generation, query execution, natural language responses, Langfuse tracing
#
# df_main contains: Component data, stock levels, sales history, demand forecasts,
# supplier info, lead times, order suggestions, and cost analysis

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from anthropic import Anthropic
from langfuse import Langfuse

from utils.dtos.rc_ml_model import RCMLModel

# Langfuse credentials (hardcoded - bypasses RapidCanvas env vars)
LANGFUSE_SECRET_KEY = "sk-lf-5126e510-c1b9-48c2-9452-e943b16ab5be"
LANGFUSE_PUBLIC_KEY = "pk-lf-a312eb3c-a112-42cb-801e-5ee483ff9128"
LANGFUSE_HOST = "https://us.cloud.langfuse.com"
from utils.notebookhelpers.helpers import Helpers

context = Helpers.getOrCreateContext(contextId="contextId", localVars=locals())


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class MTETextToSQLAgent(RCMLModel):
    """
    Agent for MTE inventory and procurement analytics.
    Analyzes df_main table containing: components, stock, sales, demand forecasts,
    suppliers, lead times, order suggestions, and costs.
    Converts natural language to SQL, executes on DuckDB, returns natural language response.
    """

    STATUS_NAMESPACE = "mte_text_to_sql_status"

    def __init__(self):
        self._anthropic_client: Optional[Anthropic] = None
        self._db_path: Optional[str] = None
        self._model = "claude-sonnet-4-20250514"
        self._current_session_id: Optional[str] = None
        self._context = None
        self._last_data_date: Optional[str] = None
        self._schema_cache: Optional[str] = None

    def load(self, artifacts: Dict[str, Any], context):
        """Initialize Anthropic client and database path."""
        self._context = context

        # Get Anthropic API key from secrets (try both naming conventions)
        anthropic_key = Helpers.get_secret(context, "anthropic_api_key")
        if not anthropic_key:
            # Fallback to environment variable
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("Anthropic API key not found in secrets or environment")

        self._anthropic_client = Anthropic(api_key=anthropic_key)

        # Download database artifact
        db_artifact = Helpers.downloadArtifacts(context, "database")
        print(f"[MTETextToSQLAgent] Downloaded artifact: {db_artifact}")

        # Find .duckdb file in artifact
        self._db_path = None
        for filename, filepath in db_artifact.items():
            if filename.endswith(".duckdb") or filepath.endswith(".duckdb"):
                self._db_path = filepath
                break

        if not self._db_path:
            raise FileNotFoundError(f"No .duckdb file found in artifact. Files: {db_artifact}")

        print(f"[MTETextToSQLAgent] Using DuckDB: {self._db_path}")

        # Get last date with data (try to find a date column dynamically)
        try:
            conn = duckdb.connect(self._db_path, read_only=True)

            # Get all tables
            tables_df = conn.execute("SHOW TABLES").fetchdf()
            tables = tables_df['name'].tolist()

            # Try to find a date column in the main table (usually df_main or first table)
            main_table = 'df_main' if 'df_main' in tables else tables[0] if tables else None

            if main_table:
                cols_df = conn.execute(f"DESCRIBE {main_table}").fetchdf()
                # Find date/timestamp columns
                date_cols = cols_df[cols_df['column_type'].str.contains('DATE|TIMESTAMP|TIME', case=False, na=False)]
                if not date_cols.empty:
                    date_col = date_cols.iloc[0]['column_name']
                    result = conn.execute(f"SELECT MAX({date_col})::DATE FROM {main_table}").fetchone()
                    if result and result[0]:
                        self._last_data_date = str(result[0])
                        print(f"[MTETextToSQLAgent] Last data date ({date_col}): {self._last_data_date}")

            conn.close()
        except Exception as e:
            print(f"[MTETextToSQLAgent] Error getting last date: {e}")

        # Cache schema on load
        self._schema_cache = self._build_schema_description()

        print("[MTETextToSQLAgent] ✅ Agent loaded successfully")
        print(f"[MTETextToSQLAgent] ✅ Langfuse configured - Host: {LANGFUSE_HOST}")

    # ------------------------- Status tracking -------------------------
    def _advance_status(self, message: str) -> None:
        """Update status message in Redis."""
        if not self._current_session_id or not self._context:
            return
        try:
            Helpers.cache_set(
                self._context,
                self.STATUS_NAMESPACE,
                self._current_session_id,
                message,
                1  # 1 hour TTL
            )
        except Exception:
            pass

    def get_status(self, context, session_id: str) -> Dict[str, Any]:
        """Get current status for a session."""
        try:
            status = Helpers.cache_get(context, self.STATUS_NAMESPACE, session_id)
            if status:
                return {"status": "success", "session_id": session_id, "message": status}
            return {"status": "success", "session_id": session_id, "message": "Ready..."}
        except Exception as e:
            return {"status": "error", "session_id": session_id, "message": str(e)}

    # ------------------------- Database helpers -------------------------
    def _get_db_connection(self):
        """Get a read-only DuckDB connection."""
        if not self._db_path:
            raise ValueError("Database path not configured")
        return duckdb.connect(self._db_path, read_only=True)

    def _build_schema_description(self) -> str:
        """Build database schema description dynamically."""
        conn = None
        try:
            conn = self._get_db_connection()

            # Get all tables
            tables_df = conn.execute("SHOW TABLES").fetchdf()
            tables = tables_df['name'].tolist()

            schema_parts = [f"""
DATABASE: MTE Inventory & Procurement Analytics (DuckDB)
Available tables: {len(tables)} tables
Last data date: {self._last_data_date or 'Unknown'}

=== TABLE DESCRIPTIONS ===

TABLE: df_main - Snapshot atual de inventário e compras:
- Component identification (Component, Cod X, Description, Group)
- Stock levels (Stock, Transit, Inspection, Total Stock)
- Sales summary (Sales 12M, Sales-M1, Sales-M2, etc.)
- Demand forecasts and lead times (LT, RP, Demand_(LT+RP))
- Supplier info (Supplier, Supp Cod)
- Order suggestions (Final_order, Model Suggestion)
- Cost analysis (Cost, Total Cost, Currency)
- Classification (ABC, IsException)

TABLE: df_vendas - Histórico detalhado de vendas por dia:
- b1_cod: Código do produto (chave para JOIN com df_produtos)
- data: Data da venda (usar para análises temporais)
- qtd_venda_nacional: Quantidade vendida no mercado nacional
- receita_venda_nacional: Receita em BRL das vendas nacionais
- qtd_venda_exportacao: Quantidade vendida para exportação
- receita_venda_exportacao: Receita em BRL das exportações
- nao_atendidos_repo: Pedidos não atendidos (reposição)
- nao_atendidos_expo: Pedidos não atendidos (exportação)

TABLE: df_produtos - Cadastro de produtos:
- b1_cod: Código do produto (chave primária)
- b1_desc: Descrição do produto
- b1_grupo: Código do grupo
- nom_grup: Nome do grupo (ex: VALVULA, SENSOR, BOBINA, etc.)
- origem: Origem (N=Nacional, I=Importado)
- curva: Classificação ABC (A, B, C, D)
- preco_venda_brl/usd/eur: Preços de venda

=== JOINS IMPORTANTES ===
- df_vendas.b1_cod = df_produtos.b1_cod (para obter descrição e grupo dos produtos vendidos)
- Para filtrar por tipo de produto (válvulas, sensores, etc), use: df_produtos.nom_grup ILIKE '%VALVULA%'

"""]

            # Get schema for each table with sample data
            for table_name in tables:
                try:
                    cols_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

                    schema_parts.append(f"TABLE: {table_name} ({count:,} rows)")
                    schema_parts.append("Columns:")
                    for _, row in cols_df.iterrows():
                        schema_parts.append(f"  - {row['column_name']}: {row['column_type']}")

                    # Add sample data for context
                    sample_df = conn.execute(f"SELECT * FROM {table_name} LIMIT 2").fetchdf()
                    if not sample_df.empty:
                        schema_parts.append("Sample data:")
                        schema_parts.append(sample_df.to_string(index=False, max_colwidth=30))
                    schema_parts.append("")
                except Exception as e:
                    schema_parts.append(f"TABLE: {table_name} (error reading schema: {e})")
                    schema_parts.append("")

            # Add general instructions (no hardcoded column names)
            schema_parts.append(f"""
=== INSTRUCTIONS ===
- Use ONLY the column names shown above in each table schema
- Do NOT assume column names - check the schema first
- For date columns, use ::DATE for comparisons
- For "today", "current", "recent": use '{self._last_data_date or "2024-12-31"}' (last available data)
- If no date specified: use last 12 months
- Use DATE_TRUNC('month', date_column) for monthly aggregations
- Use DATE_TRUNC('year', date_column) for yearly aggregations
- Limit results to 1000 rows when not specified
- Use ROUND() for decimal values (2 decimal places)

=== EXEMPLOS DE QUERIES TEMPORAIS ===

1. Vendas mensais de um tipo de produto nos últimos 12 meses:
SELECT
    DATE_TRUNC('month', v.data) AS mes,
    p.nom_grup AS grupo,
    SUM(v.qtd_venda_nacional + v.qtd_venda_exportacao) AS qtd_total,
    SUM(v.receita_venda_nacional + v.receita_venda_exportacao) AS receita_total
FROM df_vendas v
JOIN df_produtos p ON v.b1_cod = p.b1_cod
WHERE p.nom_grup ILIKE '%VALVULA%'
  AND v.data >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '12 months'
GROUP BY 1, 2
ORDER BY 1;

2. Evolução de vendas de um produto específico:
SELECT
    DATE_TRUNC('month', data) AS mes,
    SUM(qtd_venda_nacional) AS qtd_nacional,
    SUM(qtd_venda_exportacao) AS qtd_exportacao
FROM df_vendas
WHERE b1_cod = 'T1234'
GROUP BY 1
ORDER BY 1;

3. Top produtos por receita em um período:
SELECT
    p.b1_cod,
    p.b1_desc,
    p.nom_grup,
    SUM(v.receita_venda_nacional + v.receita_venda_exportacao) AS receita_total
FROM df_vendas v
JOIN df_produtos p ON v.b1_cod = p.b1_cod
WHERE v.data >= '2024-01-01'
GROUP BY 1, 2, 3
ORDER BY 4 DESC
LIMIT 20;
""")

            return "\n".join(schema_parts)

        except Exception as e:
            return f"Error building schema: {e}"
        finally:
            if conn:
                conn.close()

    def _get_schema_description(self) -> str:
        """Get cached database schema."""
        if self._schema_cache:
            return self._schema_cache
        return self._build_schema_description()

    def _execute_query(self, sql: str) -> Tuple[List[dict], Optional[str]]:
        """Execute SQL and return results as list of dicts."""
        conn = None
        try:
            conn = self._get_db_connection()
            result = conn.execute(sql).fetchdf()

            # Limit to 100 rows
            if len(result) > 100:
                result = result.head(100)

            # Convert to list of dicts
            records = result.to_dict(orient="records")

            # Convert any non-serializable types
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif hasattr(value, "isoformat"):
                        record[key] = value.isoformat()

            return records, None
        except Exception as e:
            return [], str(e)
        finally:
            if conn:
                conn.close()

    # ------------------------- SQL validation -------------------------
    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Basic SQL validation - only allow SELECT statements."""
        sql_upper = sql.upper().strip()

        # Must start with SELECT (or WITH for CTEs)
        if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
            return False, "Only SELECT queries are allowed"

        # Forbidden keywords
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "EXEC", "EXECUTE"]
        for keyword in forbidden:
            if re.search(rf"\b{keyword}\b", sql_upper):
                return False, f"Forbidden keyword: {keyword}"

        return True, None

    def _clean_sql(self, sql: str) -> str:
        """Remove markdown code blocks and clean up SQL."""
        sql = sql.strip()

        # Remove markdown code blocks
        if sql.startswith("```"):
            lines = sql.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            sql = "\n".join(lines)

        # Remove any remaining backticks
        sql = sql.strip("`").strip()

        # Remove 'sql' prefix if present
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

        # Check for truncated SQL (incomplete operators at the end)
        truncation_indicators = ["||", "AND", "OR", "WHERE", "FROM", "JOIN", ",", "("]
        sql_stripped = sql.rstrip(";").strip()
        for indicator in truncation_indicators:
            if sql_stripped.endswith(indicator):
                # SQL appears truncated, try to fix by removing incomplete part
                sql = sql_stripped.rsplit(indicator, 1)[0].strip()
                break

        # Ensure it ends with semicolon
        if sql and not sql.endswith(";"):
            sql += ";"

        return sql

    # ------------------------- LLM helpers -------------------------
    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call LLM and return response."""
        response = self._anthropic_client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question."""
        self._advance_status("Gerando query SQL...")

        schema = self._get_schema_description()

        prompt = f"""Você é um especialista em SQL para análise de inventário e compras da MTE (DuckDB).

TABELAS DISPONÍVEIS:
- df_main: Snapshot atual de inventário (estoque, demanda, fornecedores, custos)
- df_vendas: Histórico detalhado de vendas por dia (para análises temporais)
- df_produtos: Cadastro de produtos (descrição, grupo, origem, curva ABC)

IMPORTANTE:
- Valores monetários em df_main estão em USD
- Valores em df_vendas (receita) estão em BRL

{schema}

REGRAS:
1. APENAS gere SELECT statements
2. Use EXATAMENTE os nomes de colunas mostrados no schema (case-sensitive)
3. Para análises de estoque atual: use df_main
4. Para análises temporais/históricas de vendas: use df_vendas JOIN df_produtos
5. Para filtrar por tipo de produto (válvulas, sensores, etc): use df_produtos.nom_grup ILIKE '%TERMO%'
6. Para evolução mensal: use DATE_TRUNC('month', data) e GROUP BY
7. Limite a 1000 linhas quando não especificado
8. Use ROUND() para valores decimais (2 casas)
9. Ordene resultados de forma lógica (cronológica para séries temporais)
10. Para perguntas sobre "comportamento" ou "evolução", retorne dados mês a mês

PERGUNTA DO USUÁRIO: {question}

Gere APENAS a query SQL, sem explicação. A query deve ser completa e funcional:"""

        sql = self._call_llm(prompt, max_tokens=1000)
        sql = sql.strip()
        sql = self._clean_sql(sql)

        return sql

    def _generate_response(self, question: str, sql: str, results: List[dict], error: Optional[str]) -> str:
        """Generate natural language response from query results."""
        self._advance_status("Gerando resposta...")

        if error:
            context_str = f"Erro SQL: {error}"
        elif not results:
            context_str = "A query não retornou resultados."
        else:
            sample = results[:20]
            context_str = f"Query retornou {len(results)} linhas. Amostra:\n{json.dumps(sample, indent=2, default=str, ensure_ascii=False)}"

        prompt = f"""Você é um assistente de analytics de inventário e compras da MTE.

TABELAS DISPONÍVEIS:
- df_main: Snapshot atual de inventário (valores em USD)
- df_vendas: Histórico de vendas por dia (receitas em BRL)
- df_produtos: Cadastro de produtos

Responda a pergunta do usuário baseado nos resultados da query.

PERGUNTA: {question}

SQL EXECUTADO:
{sql}

RESULTADOS:
{context_str}

INSTRUÇÕES PARA RESPOSTA:
1. Forneça uma resposta clara e profissional em português
2. Para análises temporais, descreva a tendência (crescimento, queda, estabilidade)
3. Destaque os meses de pico e de baixa quando aplicável
4. Calcule variações percentuais relevantes (ex: crescimento ano a ano)
5. Se os dados forem de df_vendas, receitas estão em BRL (R$)
6. Se os dados forem de df_main, valores estão em USD (US$)
7. Seja conciso mas inclua insights relevantes
8. Se houve erro, explique o que aconteceu"""

        return self._call_llm(prompt, max_tokens=1000)

    # ------------------------- Langfuse helpers -------------------------
    def _get_langfuse_client(self):
        """Create Langfuse client directly with hardcoded credentials (not from env vars)."""
        try:
            # Create client directly with hardcoded credentials
            # This bypasses any env vars that might be set by RapidCanvas
            client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            print(f"[Langfuse] ✅ Client created with hardcoded credentials")
            print(f"[Langfuse] Public key: {LANGFUSE_PUBLIC_KEY[:20]}...")
            print(f"[Langfuse] Host: {LANGFUSE_HOST}")

            # Verify authentication
            print(f"[Langfuse] Checking authentication...")
            try:
                auth_result = client.auth_check()
                print(f"[Langfuse] ✅ Auth check result: {auth_result}")
            except Exception as auth_e:
                print(f"[Langfuse] ❌ Auth check failed: {auth_e}")

            return client
        except Exception as e:
            print(f"[Langfuse] ❌ Error creating client: {e}")
            raise

    # ------------------------- Main predict -------------------------
    def _process_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Internal method to process the query with Langfuse tracing (SDK v3 manual API).
        """
        # Get Langfuse client
        print(f"[Langfuse] Getting client...")
        langfuse = self._get_langfuse_client()

        # Create root span manually (SDK v3 style)
        print(f"[Langfuse] Creating root span...")
        try:
            root_span = langfuse.start_span(name="MTE TextToSQL Pipeline")
            print(f"[Langfuse] ✅ Root span created: {root_span}")
            print(f"[Langfuse] Root span type: {type(root_span)}")
            print(f"[Langfuse] Root span attributes: {dir(root_span)}")
        except Exception as e:
            print(f"[Langfuse] ❌ Error creating root span: {e}")
            # Fallback: process without Langfuse
            return self._process_query_without_langfuse(query, session_id)

        try:
            root_span.update(input=query, session_id=session_id)
            print(f"[Langfuse] Root span updated with input")
        except Exception as e:
            print(f"[Langfuse] ❌ Error updating root span: {e}")

        try:
            trace_id = root_span.trace_id
            print(f"[Langfuse] ✅ Trace ID: {trace_id}")
        except AttributeError:
            print(f"[Langfuse] ⚠️ root_span has no trace_id, trying .id")
            try:
                trace_id = root_span.id
                print(f"[Langfuse] ✅ Span ID: {trace_id}")
            except Exception as e:
                print(f"[Langfuse] ❌ Error getting ID: {e}")
                trace_id = session_id

        try:
            self._advance_status("Entendendo sua pergunta...")

            # 1. Generate SQL (with child span)
            print(f"[Langfuse] Creating SQLGeneratorAgent span...")
            try:
                sql_span = root_span.start_span(name="SQLGeneratorAgent")
                sql_span.update(input=query)
                print(f"[Langfuse] ✅ SQLGeneratorAgent span created")
            except Exception as e:
                print(f"[Langfuse] ❌ Error creating SQL span: {e}")
                sql_span = None

            sql = self._generate_sql(query)

            if sql_span:
                try:
                    sql_span.update(output=sql)
                    sql_span.end()
                    print(f"[Langfuse] ✅ SQLGeneratorAgent span ended")
                except Exception as e:
                    print(f"[Langfuse] ❌ Error ending SQL span: {e}")

            print(f"[MTETextToSQLAgent] Generated SQL: {sql}")

            # 2. Validate SQL
            is_valid, validation_error = self._validate_sql(sql)
            if not is_valid:
                self._advance_status("Query rejeitada")
                try:
                    root_span.update(output=f"Query rejeitada: {validation_error}")
                    root_span.end()
                except Exception as e:
                    print(f"[Langfuse] ❌ Error ending root span: {e}")
                return {
                    "response": f"Query rejeitada por motivos de segurança: {validation_error}",
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "sql": sql
                }

            self._advance_status("Executando query...")

            # 3. Execute query (with child span)
            print(f"[Langfuse] Creating SQLExecution span...")
            try:
                exec_span = root_span.start_span(name="SQLExecution")
                exec_span.update(input=sql)
                print(f"[Langfuse] ✅ SQLExecution span created")
            except Exception as e:
                print(f"[Langfuse] ❌ Error creating exec span: {e}")
                exec_span = None

            results, error = self._execute_query(sql)

            if exec_span:
                try:
                    exec_span.update(output={"row_count": len(results) if results else 0, "error": error})
                    exec_span.end()
                    print(f"[Langfuse] ✅ SQLExecution span ended")
                except Exception as e:
                    print(f"[Langfuse] ❌ Error ending exec span: {e}")

            # 4. Generate response (with child span)
            print(f"[Langfuse] Creating AnswerGeneratorAgent span...")
            try:
                answer_span = root_span.start_span(name="AnswerGeneratorAgent")
                answer_span.update(input={"question": query, "sql": sql, "row_count": len(results) if results else 0})
                print(f"[Langfuse] ✅ AnswerGeneratorAgent span created")
            except Exception as e:
                print(f"[Langfuse] ❌ Error creating answer span: {e}")
                answer_span = None

            response_text = self._generate_response(query, sql, results, error)

            if answer_span:
                try:
                    answer_span.update(output=response_text)
                    answer_span.end()
                    print(f"[Langfuse] ✅ AnswerGeneratorAgent span ended")
                except Exception as e:
                    print(f"[Langfuse] ❌ Error ending answer span: {e}")

            self._advance_status("Completo")

            # Update root span with final output and end it
            try:
                root_span.update(output=response_text)
                root_span.end()
                print(f"[Langfuse] ✅ Root span ended")
            except Exception as e:
                print(f"[Langfuse] ❌ Error ending root span: {e}")

            return {
                "response": response_text,
                "session_id": session_id,
                "trace_id": trace_id,
                "sql": sql,
                "row_count": len(results) if results else 0
            }

        except Exception as e:
            # End root span on error
            print(f"[Langfuse] ❌ Exception in pipeline: {e}")
            try:
                root_span.update(output=f"Error: {str(e)}")
                root_span.end()
            except Exception as e2:
                print(f"[Langfuse] ❌ Error ending root span after exception: {e2}")
            raise

        finally:
            # Always flush to send traces to Langfuse
            print(f"[Langfuse] Flushing...")
            try:
                result = langfuse.flush()
                print(f"[Langfuse] ✅ Flush completed: {result}")
            except Exception as e:
                print(f"[Langfuse] ❌ Error flushing: {e}")

    def _process_query_without_langfuse(self, query: str, session_id: str) -> Dict[str, Any]:
        """Fallback method without Langfuse tracing."""
        print(f"[MTETextToSQLAgent] Processing without Langfuse...")
        self._advance_status("Entendendo sua pergunta...")

        sql = self._generate_sql(query)
        print(f"[MTETextToSQLAgent] Generated SQL: {sql}")

        is_valid, validation_error = self._validate_sql(sql)
        if not is_valid:
            self._advance_status("Query rejeitada")
            return {
                "response": f"Query rejeitada por motivos de segurança: {validation_error}",
                "session_id": session_id,
                "trace_id": None,
                "sql": sql
            }

        self._advance_status("Executando query...")
        results, error = self._execute_query(sql)

        response_text = self._generate_response(query, sql, results, error)
        self._advance_status("Completo")

        return {
            "response": response_text,
            "session_id": session_id,
            "trace_id": None,
            "sql": sql,
            "row_count": len(results) if results else 0
        }

    def predict(self, model_input: Dict[str, Any], context) -> List:
        """
        Main inference method.

        Input formats:
          - {"query": "Your question"} - normal query
          - {"op": "get_status", "session_id": "..."} - get status

        Returns:
          - [response_text, session_id, trace_id]
        """
        self._context = context

        # Handle operations
        op = str(model_input.get("op", "")).strip().lower()

        if op in {"get_status", "status"}:
            session_id = model_input.get("session_id")
            if not session_id:
                return [{"status": "error", "message": "session_id is required"}, None, None]
            return [self.get_status(context, str(session_id)), session_id, None]

        # Normal query
        query = model_input.get("query") or model_input.get("user_query") or model_input.get("message")
        if not query:
            return ["Erro: Nenhuma pergunta fornecida.", None, None]

        # Generate session ID for status tracking
        session_id = model_input.get("session_id") or str(uuid.uuid4())
        self._current_session_id = session_id

        try:
            # Process query with Langfuse tracing
            result = self._process_query(query, session_id)

            return [result["response"], result["session_id"], result.get("trace_id")]

        except Exception as e:
            self._advance_status("Erro ocorreu")
            return [f"Erro ao processar sua pergunta: {str(e)}", session_id, None]
        finally:
            self._current_session_id = None


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Register as an RC ML Model
model = MTETextToSQLAgent()
model.load({}, context)

Helpers.save_output_rc_ml_model(
    context=context,
    model_name="mte_text_to_sql",
    model_obj=MTETextToSQLAgent,
    artifacts={}
)
Helpers.save(context)
print("[MTETextToSQLAgent] Registered successfully.")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Example usage (uncomment to test locally)
# model_input = {
#     "query": "Quais são os 10 componentes com maior valor de estoque?"
# }
# response_text, session_id, trace_id = model.predict(model_input, context)
# print(f"Response: {response_text}")
# print(f"Session ID: {session_id}")
# print(f"Langfuse Trace ID: {trace_id}")  # Use this for feedback tracking
#
# Other example queries:
# - "Qual o total de vendas nos últimos 12 meses por grupo?"
# - "Quais componentes têm estoque zerado mas demanda positiva?"
# - "Qual o custo total das sugestões de compra por fornecedor?"
# - "Quais componentes classe A têm lead time maior que 30 dias?"