use std::sync::Arc;

use datafusion::catalog::TableProvider;
use sail_common_datafusion::catalog::{TableKind, TableStatus};
use crate::error::{CatalogError, CatalogResult};
use crate::manager::CatalogManager;
use crate::provider::{CreateTableOptions, DropTableOptions};
use crate::utils::match_pattern;

impl CatalogManager {
    fn cache_key_for_table_status(status: &TableStatus) -> String {
        match &status.kind {
            TableKind::TemporaryView { .. } => format!("temporary::{}", status.name),
            TableKind::GlobalTemporaryView { database, .. } => {
                format!(
                    "global_temporary::{}::{}",
                    database.join("."),
                    status.name
                )
            }
            TableKind::Table {
                catalog, database, ..
            }
            | TableKind::View {
                catalog, database, ..
            } => format!(
                "object::{}::{}::{}",
                catalog,
                database.join("."),
                status.name
            ),
        }
    }

    async fn resolve_cache_key<T: AsRef<str>>(&self, table: &[T]) -> CatalogResult<String> {
        let table = self.get_table_or_view(table).await?;
        Ok(Self::cache_key_for_table_status(&table))
    }

    pub async fn cache_key_for_table<T: AsRef<str>>(&self, table: &[T]) -> CatalogResult<String> {
        self.resolve_cache_key(table).await
    }

    pub async fn cache_relation_id_for_table<T: AsRef<str>>(
        &self,
        table: &[T],
    ) -> CatalogResult<Option<String>> {
        let key = self.resolve_cache_key(table).await?;
        let state = self.state()?;
        Ok(state.cached_table_relations.get(&key).cloned())
    }

    pub async fn is_cached_table<T: AsRef<str>>(&self, table: &[T]) -> CatalogResult<bool> {
        let key = self.resolve_cache_key(table).await?;
        let state = self.state()?;
        Ok(state.cached_tables.contains(&key))
    }

    pub async fn is_lazy_cached_table<T: AsRef<str>>(&self, table: &[T]) -> CatalogResult<bool> {
        let key = self.resolve_cache_key(table).await?;
        let state = self.state()?;
        Ok(state.lazy_cached_tables.contains(&key))
    }

    pub async fn is_disk_cached_table<T: AsRef<str>>(&self, table: &[T]) -> CatalogResult<bool> {
        let key = self.resolve_cache_key(table).await?;
        let state = self.state()?;
        Ok(state.disk_cached_tables.contains(&key))
    }

    pub async fn cache_table<T: AsRef<str>>(
        &self,
        table: &[T],
        lazy: bool,
        disk_backed: bool,
    ) -> CatalogResult<()> {
        let key = self.resolve_cache_key(table).await?;
        let mut state = self.state()?;
        state.cached_tables.insert(key.clone());
        if lazy {
            state.lazy_cached_tables.insert(key.clone());
        } else {
            state.lazy_cached_tables.remove(&key);
        }
        if disk_backed {
            state.disk_cached_tables.insert(key);
        } else {
            state.disk_cached_tables.remove(&key);
        }
        Ok(())
    }

    pub async fn set_cached_table_relation<T: AsRef<str>>(
        &self,
        table: &[T],
        relation_id: String,
    ) -> CatalogResult<()> {
        let key = self.resolve_cache_key(table).await?;
        let mut state = self.state()?;
        state.cached_tables.insert(key.clone());
        state.lazy_cached_tables.remove(&key);
        state.cached_table_relations.insert(key, relation_id);
        Ok(())
    }

    pub fn set_cached_table_provider(
        &self,
        relation_id: String,
        provider: Arc<dyn TableProvider>,
    ) -> CatalogResult<()> {
        let mut state = self.state()?;
        state.cached_table_providers.insert(relation_id, provider);
        Ok(())
    }

    pub fn cached_table_provider<T: AsRef<str>>(
        &self,
        relation_id: T,
    ) -> CatalogResult<Option<Arc<dyn TableProvider>>> {
        let state = self.state()?;
        Ok(state
            .cached_table_providers
            .get(relation_id.as_ref())
            .cloned())
    }

    pub fn remove_cached_table_provider<T: AsRef<str>>(&self, relation_id: T) -> CatalogResult<()> {
        let mut state = self.state()?;
        state.cached_table_providers.remove(relation_id.as_ref());
        Ok(())
    }

    pub async fn uncache_table<T: AsRef<str>>(
        &self,
        table: &[T],
        if_exists: bool,
    ) -> CatalogResult<Option<String>> {
        let key = match self.resolve_cache_key(table).await {
            Ok(key) => key,
            Err(CatalogError::NotFound(_, _)) if if_exists => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut state = self.state()?;
        state.cached_tables.remove(&key);
        state.lazy_cached_tables.remove(&key);
        state.disk_cached_tables.remove(&key);
        let relation_id = state.cached_table_relations.remove(&key);
        if let Some(relation_id) = relation_id.as_ref() {
            state.cached_table_providers.remove(relation_id);
        }
        Ok(relation_id)
    }

    pub async fn refresh_table<T: AsRef<str>>(&self, table: &[T]) -> CatalogResult<Option<String>> {
        // Spark refresh invalidates table cache entries.
        self.uncache_table(table, true).await
    }

    pub fn clear_cached_tables(&self) -> CatalogResult<Vec<String>> {
        let mut state = self.state()?;
        state.cached_tables.clear();
        state.lazy_cached_tables.clear();
        state.disk_cached_tables.clear();
        state.cached_table_providers.clear();
        let relations = state
            .cached_table_relations
            .drain()
            .map(|(_, relation_id)| relation_id)
            .collect();
        Ok(relations)
    }

    pub async fn create_table<T: AsRef<str>>(
        &self,
        table: &[T],
        options: CreateTableOptions,
    ) -> CatalogResult<TableStatus> {
        let (provider, database, table) = self.resolve_object(table)?;
        provider.create_table(&database, &table, options).await
    }

    pub async fn get_table<T: AsRef<str>>(&self, table: &[T]) -> CatalogResult<TableStatus> {
        let (provider, database, table) = self.resolve_object(table)?;
        provider.get_table(&database, &table).await
    }

    pub async fn list_tables<T: AsRef<str>>(
        &self,
        database: &[T],
        pattern: Option<&str>,
    ) -> CatalogResult<Vec<TableStatus>> {
        let (provider, database) = if database.is_empty() {
            self.resolve_default_database()?
        } else {
            self.resolve_database(database)?
        };
        Ok(provider
            .list_tables(&database)
            .await?
            .into_iter()
            .filter(|x| match_pattern(&x.name, pattern))
            .collect())
    }

    pub async fn list_tables_and_temporary_views<T: AsRef<str>>(
        &self,
        database: &[T],
        pattern: Option<&str>,
    ) -> CatalogResult<Vec<TableStatus>> {
        // Spark *global* temporary views should be put in the "global temporary" database, and they will be
        // included in the output if the database name matches.
        let mut output = if self.state()?.is_global_temporary_view_database(database) {
            self.list_global_temporary_views(pattern).await?
        } else {
            self.list_tables(database, pattern).await?
        };
        // Spark (local) temporary views are session-scoped and are not associated with a catalog.
        // We should include the temporary views in the output.
        output.extend(self.list_temporary_views(pattern).await?);
        Ok(output)
    }

    pub async fn drop_table<T: AsRef<str>>(
        &self,
        table: &[T],
        options: DropTableOptions,
    ) -> CatalogResult<()> {
        let (provider, database, table) = self.resolve_object(table)?;
        provider.drop_table(&database, &table, options).await
    }

    pub async fn get_table_or_view<T: AsRef<str>>(
        &self,
        reference: &[T],
    ) -> CatalogResult<TableStatus> {
        if let [name] = reference {
            match self.get_temporary_view(name.as_ref()).await {
                Ok(x) => return Ok(x),
                Err(CatalogError::NotFound(_, _)) => {}
                Err(e) => return Err(e),
            }
        }
        if let [x @ .., name] = reference {
            if self.state()?.is_global_temporary_view_database(x) {
                return self.get_global_temporary_view(name.as_ref()).await;
            }
        }
        match self.get_table(reference).await {
            Ok(x) => return Ok(x),
            Err(CatalogError::NotFound(_, _)) => {}
            Err(e) => return Err(e),
        }
        self.get_view(reference).await
    }
}
