use std::sync::Arc;

use datafusion_common::ScalarValue;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::select_expr::SelectExpr;
use datafusion_expr::{col, lit, Expr, LogicalPlan, LogicalPlanBuilder, ScalarUDF};
use rand::{rng, Rng};
use sail_common::spec;
use sail_function::scalar::array::spark_sequence::SparkSequence;
use sail_function::scalar::math::rand_poisson::RandPoisson;
use sail_function::scalar::math::random::Random;

use crate::error::{PlanError, PlanResult};
use crate::resolver::state::PlanResolverState;
use crate::resolver::PlanResolver;

impl PlanResolver<'_> {
    pub(super) async fn resolve_query_sample(
        &self,
        sample: spec::Sample,
        state: &mut PlanResolverState,
    ) -> PlanResult<LogicalPlan> {
        let spec::Sample {
            input,
            lower_bound,
            upper_bound,
            with_replacement,
            seed,
            ..
        } = sample;
        if lower_bound >= upper_bound {
            return Err(PlanError::invalid(format!(
                "invalid sample bounds: [{lower_bound}, {upper_bound})"
            )));
        }
        // if defined seed use these values otherwise use random seed
        // to generate the random values in with_replacement mode, in lambda value
        let seed: i64 = seed.unwrap_or_else(|| {
            let mut rng = rng();
            rng.random::<i64>()
        });

        let input: LogicalPlan = self
            .resolve_query_plan_with_hidden_fields(*input, state)
            .await?;
        let rand_column_name: String = state.register_field_name("rand_value");
        let rand_expr: Expr = if with_replacement {
            Expr::ScalarFunction(ScalarFunction {
                func: Arc::new(ScalarUDF::from(RandPoisson::new())),
                args: vec![
                    Expr::Literal(ScalarValue::Float64(Some(upper_bound)), None),
                    Expr::Literal(ScalarValue::Int64(Some(seed)), None),
                ],
            })
            .alias(&rand_column_name)
        } else {
            Expr::ScalarFunction(ScalarFunction {
                func: Arc::new(ScalarUDF::from(Random::new())),
                args: vec![Expr::Literal(ScalarValue::Int64(Some(seed)), None)],
            })
            .alias(&rand_column_name)
        };
        let init_exprs: Vec<Expr> = input
            .schema()
            .columns()
            .iter()
            .map(|col| Expr::Column(col.clone()))
            .collect();
        let mut all_exprs: Vec<Expr> = init_exprs.clone();
        all_exprs.push(rand_expr);
        let plan_with_rand: LogicalPlan = LogicalPlanBuilder::from(input)
            .project(all_exprs)?
            .build()?;

        if with_replacement {
            Self::resolve_sample_with_replacement(
                plan_with_rand,
                &rand_column_name,
                init_exprs,
                state,
            )
        } else {
            Self::resolve_sample_without_replacement(
                plan_with_rand,
                &rand_column_name,
                lower_bound,
                upper_bound,
                init_exprs,
            )
        }
    }

    /// Bernoulli sampling - filter rows where random value falls in [lower, upper)
    fn resolve_sample_without_replacement(
        plan_with_rand: LogicalPlan,
        rand_column_name: &str,
        lower_bound: f64,
        upper_bound: f64,
        init_exprs: Vec<Expr>,
    ) -> PlanResult<LogicalPlan> {
        let plan = LogicalPlanBuilder::from(plan_with_rand)
            .filter(col(rand_column_name).lt(lit(upper_bound)))?
            .filter(col(rand_column_name).gt_eq(lit(lower_bound)))?
            .build()?;
        let plan = LogicalPlanBuilder::from(plan)
            .project(init_exprs)?
            .build()?;
        Ok(plan)
    }

    /// Poisson sampling - replicate rows based on Poisson distribution
    fn resolve_sample_with_replacement(
        plan_with_rand: LogicalPlan,
        rand_column_name: &str,
        init_exprs: Vec<Expr>,
        state: &mut PlanResolverState,
    ) -> PlanResult<LogicalPlan> {
        let init_exprs_aux: Vec<Expr> = plan_with_rand
            .schema()
            .columns()
            .iter()
            .map(|col| Expr::Column(col.clone()))
            .collect();
        let array_column_name: String = state.register_field_name("array_value");
        let arr_expr: Expr = Expr::ScalarFunction(ScalarFunction {
            func: Arc::new(ScalarUDF::from(SparkSequence::new())),
            args: vec![
                Expr::Literal(ScalarValue::Int64(Some(1)), None),
                col(rand_column_name),
            ],
        })
        .alias(&array_column_name);
        let plan = LogicalPlanBuilder::from(plan_with_rand)
            .project(
                init_exprs_aux
                    .into_iter()
                    .chain(vec![arr_expr])
                    .map(Into::into)
                    .collect::<Vec<SelectExpr>>(),
            )?
            .build()?;
        let plan = LogicalPlanBuilder::from(plan)
            .unnest_column(array_column_name)?
            .build()?;
        let plan = LogicalPlanBuilder::from(plan)
            .project(
                init_exprs
                    .into_iter()
                    .map(Into::into)
                    .collect::<Vec<SelectExpr>>(),
            )?
            .build()?;
        Ok(plan)
    }
}
