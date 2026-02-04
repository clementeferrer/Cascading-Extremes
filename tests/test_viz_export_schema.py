from cascades.viz_export.schema import EventRecord, MetricsRecord, RunMeta


def test_run_meta_schema():
    meta = RunMeta(
        run_id="test_run",
        created_at="2026-02-03T00:00:00Z",
        source="real",
        assets=["BTC-USD", "ETH-USD", "BNB-USD"],
        freq="1h",
        threshold={"tau": 0.98, "model": "directional_quantile_mlp"},
        model_checkpoint=None,
        config_hash="sha256:abc",
    )
    assert meta.run_id == "test_run"


def test_event_record_schema():
    event = EventRecord(
        id=1,
        t=10.0,
        w=[0.2, 0.3, 0.5],
        mag=1.5,
        u_tau=1.2,
        asset="BTC-USD",
        intensity=0.8,
        parent_id=None,
        is_real=True,
    )
    assert event.asset == "BTC-USD"


def test_metrics_record_schema_alias():
    record = MetricsRecord.model_validate(
        {
            "t": 10.0,
            "lambda": 0.9,
            "psi": 0.3,
            "mu": 0.6,
            "mean_mag": 1.4,
            "event_rate": 0.5,
            "per_asset_counts": {"BTC-USD": 2},
            "direction_density_bin": None,
        }
    )
    assert record.lambda_ == 0.9
