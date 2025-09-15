# complete_model_promotion_guide.py
# Complete model promotion and deployment workflow for multi-task crypto prediction models

from datetime import datetime
import logging

from model_training import ModelTrainingPipeline, MLflowModelRegistry
from inference_feature_pipeline import MultiTaskModelPredictor, INFERENCE_TASKS

class ModelPromotion:
    def __init__(self, db_config):
        self.db_config = db_config

    def promote_models_to_staging(self, training_results):
        """Complete model promotion workflow"""

        trainer = ModelTrainingPipeline(self.db_config)
        registry = MLflowModelRegistry()
        
        # STEP 1: After training, get your training results
        print("=" * 50)
        print("STEP 1: Training Results Analysis")
        print("=" * 50)
        
        # STEP 2: Automatic promotion to staging (best model per task)
        print("STEP 2: Promoting Best Models to Staging")
        print("=" * 50)
        
        promoted_models = trainer.promote_best_models_to_staging(training_results)
        
        for model in promoted_models:
            print(f"STAGING: {model['registered_name']}")
            print(f"   Model: {model['model_type']}, Metric Name: {model['metric_name']}, Metrics Value: {model['metric_value']:.6f}")
            print(f"   Version: {model['version']}")
        
        # # STEP 3: Manual promotion to production (after validation)
        # print("\nSTEP 3: Manual Promotion to Production")
        # print("=" * 50)
        
        # # You would manually review and promote models to production
        # production_candidates = [
        #     ('crypto_lightgbm_return_4step_BTCUSDT', '2'),
        #     ('crypto_xgboost_direction_4step_BTCUSDT', '1'),
        #     ('crypto_lightgbm_volatility_4step_BTCUSDT', '1')
        # ]
        
        # for model_name, version in production_candidates:
        #     try:
        #         registry.promote_model_to_production(model_name, version)
                
        #         # Add description
        #         registry.add_model_description(
        #             model_name, version, 
        #             f"Production model promoted on {datetime.now().strftime('%Y-%m-%d')} after validation"
        #         )
                
        #         print(f"PRODUCTION: {model_name} v{version}")
                
        #     except Exception as e:
        #         print(f"Failed to promote {model_name}: {e}")
        
        return promoted_models

    def deploy_inference_service():
        """Deploy the inference service with production models"""
        
        redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
        
        # STEP 4: Initialize predictor and load production models
        print("STEP 4: Loading Production Models for Inference")
        print("=" * 50)

        predictor = MultiTaskModelPredictor(self.db_config, redis_config)

        # Load production models for specific symbols and tasks
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        priority_tasks = ['return_4step', 'direction_4step', 'volatility_4step']  # Most important tasks
        
        loaded, failed = predictor.load_production_models(
            symbols=symbols,
            tasks=priority_tasks,  # Load only priority tasks initially
            model_types=['lightgbm', 'xgboost']
        )
        
        print(f"Loaded {loaded} models, {failed} failed")
        
        # Show model status
        status = predictor.get_model_status()
        print(f"\nModel Status:")
        print(f"  Total models loaded: {status['total_models']}")
        print(f"  Production models: {status['models_by_stage']['Production']}")
        print(f"  Staging models: {status['models_by_stage']['Staging']}")
        print(f"  Models by symbol: {status['models_by_symbol']}")
        
        # STEP 5: Test predictions
        print("\nSTEP 5: Testing Predictions")
        print("=" * 50)
        
        # Test prediction for BTCUSDT
        try:
            predictions = predictor.predict_all_tasks('BTCUSDT')
            
            if 'error' in predictions:
                print(f"Prediction failed: {predictions['error']}")
            else:
                print(f"Predictions for BTCUSDT:")
                print(f"  Features used: {predictions['features_count']}")
                
                # Show ensemble predictions
                for task, pred in predictions['ensemble_predictions'].items():
                    task_desc = INFERENCE_TASKS.get(task, {}).get('description', task)
                    
                    if 'prediction' in pred:  # Regression
                        print(f"  {task_desc}: {pred['prediction']:.6f} (confidence: {pred['confidence']:.2f})")
                    elif 'class_prediction' in pred:  # Classification
                        if 'probability' in pred:  # Binary
                            direction = "UP" if pred['class_prediction'] == 1 else "DOWN"
                            print(f"  {task_desc}: {direction} (prob: {pred['probability']:.3f})")
                        else:  # Multi-class
                            classes = ['strong_down', 'down', 'neutral', 'up', 'strong_up']
                            predicted_class = classes[pred['class_prediction']] if pred['class_prediction'] < len(classes) else 'unknown'
                            print(f"  {task_desc}: {predicted_class.upper()} (conf: {pred['confidence']:.3f})")
        
        except Exception as e:
            print(f"Test prediction failed: {e}")
        
        return predictor

    def compare_and_manage_models():
        """Model comparison and management utilities"""
        
        registry = MLflowModelRegistry()
        
        print("STEP 6: Model Comparison and Management")
        print("=" * 50)
        
        # Compare model versions
        model_name = "crypto_lightgbm_return_4step_BTCUSDT"
        
        try:
            comparison = registry.compare_model_versions(model_name, "1", "2")
            
            print(f"Comparing {model_name}:")
            v1_rmse = comparison['version_1']['metrics'].get('rmse', 'N/A')
            v2_rmse = comparison['version_2']['metrics'].get('rmse', 'N/A')
            
            print(f"  Version 1: RMSE = {v1_rmse}")
            print(f"  Version 2: RMSE = {v2_rmse}")
            
            if isinstance(v1_rmse, float) and isinstance(v2_rmse, float):
                better_version = "Version 2" if v2_rmse < v1_rmse else "Version 1"
                print(f"  {better_version} performs better")
        
        except Exception as e:
            print(f"Model comparison failed: {e}")
        
        # Get current production models
        print("\nCurrent Production Models:")
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            for task in ['return_4step', 'direction_4step']:
                for model_type in ['lightgbm', 'xgboost']:
                    model_name = f"crypto_{model_type}_{task}_{symbol}"
                    
                    try:
                        prod_model = registry.get_production_model(model_name)
                        if prod_model:
                            print(f"  {model_name} v{prod_model.version}")
                        else:
                            print(f"  {model_name} - No production version")
                    except:
                        print(f"  {model_name} - Not found")

    def automated_promotion_with_validation():
        """Automated promotion with performance validation"""
        
        registry = MLflowModelRegistry()
        
        print("STEP 7: Automated Model Promotion with Validation")
        print("=" * 50)
        
        # Define promotion criteria
        promotion_criteria = {
            'regression': {
                'min_r2': 0.05,      # Minimum R-squared
                'max_rmse': 0.05,    # Maximum RMSE
                'min_directional_accuracy': 0.52  # Better than random
            },
            'classification_binary': {
                'min_accuracy': 0.53,     # Better than random (50%)
                'min_f1': 0.50,
                'min_precision': 0.50
            },
            'classification_multi': {
                'min_accuracy': 0.22,     # Better than random (20% for 5 classes)
                'min_f1': 0.20
            }
        }
        
        # Get all staging models and evaluate for production promotion
        candidate_models = [
            ('crypto_lightgbm_return_4step_BTCUSDT', '2', {'rmse': 0.0234, 'r2': 0.156, 'directional_accuracy': 0.534}),
            ('crypto_xgboost_direction_4step_BTCUSDT', '1', {'accuracy': 0.567, 'f1': 0.543, 'precision': 0.552}),
            ('crypto_lightgbm_volatility_4step_ETHUSDT', '1', {'rmse': 0.0456, 'r2': 0.089, 'directional_accuracy': 0.487})
        ]
        
        for model_name, version, metrics in candidate_models:
            # Determine task type from model name
            if 'return_' in model_name or 'volatility_' in model_name:
                task_type = 'regression'
                criteria = promotion_criteria['regression']
                
                meets_criteria = (
                    metrics.get('r2', 0) >= criteria['min_r2'] and
                    metrics.get('rmse', float('inf')) <= criteria['max_rmse'] and
                    metrics.get('directional_accuracy', 0) >= criteria['min_directional_accuracy']
                )
                
            elif 'direction_multi_' in model_name or 'vol_regime_' in model_name:
                task_type = 'classification_multi'
                criteria = promotion_criteria['classification_multi']
                
                meets_criteria = (
                    metrics.get('accuracy', 0) >= criteria['min_accuracy'] and
                    metrics.get('f1', 0) >= criteria['min_f1']
                )
                
            else:  # binary classification
                task_type = 'classification_binary' 
                criteria = promotion_criteria['classification_binary']
                
                meets_criteria = (
                    metrics.get('accuracy', 0) >= criteria['min_accuracy'] and
                    metrics.get('f1', 0) >= criteria['min_f1'] and
                    metrics.get('precision', 0) >= criteria['min_precision']
                )
            
            print(f"Evaluating {model_name} v{version} ({task_type}):")
            print(f"  Metrics: {metrics}")
            
            if meets_criteria:
                try:
                    registry.promote_model_to_production(model_name, version)
                    registry.add_model_description(
                        model_name, version,
                        f"Auto-promoted to production - meets all criteria for {task_type}"
                    )
                    print(f"  PROMOTED to production")
                    
                except Exception as e:
                    print(f"  PROMOTION FAILED: {e}")
            else:
                print(f"  NOT PROMOTED - doesn't meet criteria")

    def archive_old_models():
        """Archive outdated model versions to keep registry clean"""
        
        registry = MLflowModelRegistry()
        
        print("STEP 8: Archiving Old Model Versions")
        print("=" * 50)
        
        # Example: Archive old versions of models
        models_to_archive = [
            ('crypto_lightgbm_return_4step_BTCUSDT', '1'),  # Older version
            ('crypto_xgboost_direction_4step_BTCUSDT', '1'),
        ]
        
        for model_name, version in models_to_archive:
            try:
                registry.archive_model_version(model_name, version)
                print(f"Archived {model_name} v{version}")
            except Exception as e:
                print(f"Failed to archive {model_name} v{version}: {e}")

    def monitor_model_performance():
        """Monitor production model performance over time"""
        
        print("STEP 9: Model Performance Monitoring")
        print("=" * 50)
        
        # This would typically connect to your monitoring system
        # For demo, we'll simulate performance checks
        production_models = [
            ('crypto_lightgbm_return_4step_BTCUSDT', {'current_rmse': 0.0245, 'baseline_rmse': 0.0234}),
            ('crypto_xgboost_direction_4step_BTCUSDT', {'current_accuracy': 0.545, 'baseline_accuracy': 0.567}),
        ]
        
        performance_alerts = []
        
        for model_name, metrics in production_models:
            if 'return_' in model_name or 'volatility_' in model_name:
                # Regression model - check RMSE degradation
                current = metrics['current_rmse']
                baseline = metrics['baseline_rmse']
                degradation = (current - baseline) / baseline * 100
                
                if degradation > 10:  # More than 10% degradation
                    performance_alerts.append({
                        'model': model_name,
                        'issue': f'RMSE degraded by {degradation:.1f}%',
                        'action': 'Consider retraining or rollback'
                    })
                    print(f"ALERT: {model_name} - RMSE degraded by {degradation:.1f}%")
                else:
                    print(f"{model_name} - Performance stable")
                    
            else:
                # Classification model - check accuracy degradation
                current = metrics['current_accuracy']
                baseline = metrics['baseline_accuracy']
                degradation = (baseline - current) / baseline * 100
                
                if degradation > 5:  # More than 5% accuracy drop
                    performance_alerts.append({
                        'model': model_name,
                        'issue': f'Accuracy dropped by {degradation:.1f}%',
                        'action': 'Consider retraining or rollback'
                    })
                    print(f"ALERT: {model_name} - Accuracy dropped by {degradation:.1f}%")
                else:
                    print(f"{model_name} - Performance stable")
        
        return performance_alerts

    def complete_deployment_workflow():
        """Complete end-to-end deployment workflow"""
        
        print("COMPLETE DEPLOYMENT WORKFLOW")
        print("=" * 60)
        
        workflow_steps = [
            "1. Train models with multi-task pipeline",
            "2. Validate data quality and model performance", 
            "3. Promote best models to staging automatically",
            "4. Manual review and testing of staging models",
            "5. Promote validated models to production",
            "6. Deploy inference service with production models",
            "7. Monitor performance and set up alerts",
            "8. Archive old model versions",
            "9. Set up automated retraining pipeline"
        ]
        
        for step in workflow_steps:
            print(f"  {step}")
        
        print(f"\nDEPLOYMENT CHECKLIST:")
        checklist = [
            "Data validation pipeline active",
            "MLflow model registry configured", 
            "Production models promoted and tested",
            "FastAPI inference service deployed",
            "Model performance monitoring enabled",
            "Automated promotion criteria defined",
            "Rollback procedures documented",
            "Retraining schedule established"
        ]
        
        for item in checklist:
            print(f"  [ ] {item}")

# def main():
#     """Main deployment workflow execution"""
    
#     print("CRYPTO MODEL PROMOTION & DEPLOYMENT PIPELINE")
#     print("=" * 60)
    
#     try:
#         # Step 1: Promote models to staging
#         promoted_models = promote_models_to_staging(training_results)

#         # Step 2: Deploy inference service
#         predictor = deploy_inference_service()
        
#         # Step 3: Compare and manage models
#         compare_and_manage_models()
        
#         # Step 4: Automated promotion with validation
#         automated_promotion_with_validation()
        
#         # Step 5: Archive old models
#         archive_old_models()
        
#         # Step 6: Monitor performance
#         alerts = monitor_model_performance()
        
#         # Step 7: Show complete workflow
#         complete_deployment_workflow()
        
#         print("\n" + "=" * 60)
#         print("DEPLOYMENT COMPLETE")
#         print("=" * 60)
#         print("Your models are now ready for:")
#         print("  - Real-time price predictions (15min, 1h, 4h)")
#         print("  - Direction classification (binary & multi-class)")
#         print("  - Volatility forecasting & regime detection")
#         print("  - Automated model management & promotion")
        
#         if alerts:
#             print(f"\nPerformance alerts detected: {len(alerts)}")
#             for alert in alerts:
#                 print(f"    - {alert['model']}: {alert['issue']}")
        
#         print(f"\nNext steps:")
#         print("  1. Start FastAPI service: uvicorn production_api:app --host 0.0.0.0 --port 8000")
#         print("  2. Test predictions: GET /predict/BTCUSDT")
#         print("  3. Set up monitoring dashboards")
#         print("  4. Configure automated retraining")
        
#     except Exception as e:
#         print(f"Deployment failed: {e}")
#         import traceback
#         traceback.print_exc()

# # Usage patterns for different scenarios:

# def quick_promotion_example():
#     """Quick example of promoting specific models"""
    
#     registry = MLflowModelRegistry()
    
#     # Promote specific model to staging
#     model_name = "crypto_lightgbm_return_4step_BTCUSDT"
#     version = registry.promote_model_to_staging(model_name)
#     print(f"Promoted {model_name} v{version} to staging")
    
#     # Then promote to production after validation
#     registry.promote_model_to_production(model_name, version)
#     print(f"Promoted {model_name} v{version} to production")

# def bulk_promotion_example():
#     """Promote all good models at once"""
    
#     trainer = ModelTrainingPipeline({})
    
#     # After training, this automatically promotes best models to staging
#     training_results = {}  # Your actual training results
#     promoted = trainer.promote_best_models_to_staging(training_results)
    
#     print(f"Auto-promoted {len(promoted)} models to staging")

# def load_specific_models_example():
#     """Load only specific models for inference"""
    
#     predictor = MultiTaskModelPredictor({}, {})
    
#     # Load only price prediction models
#     predictor.load_production_models(
#         symbols=['BTCUSDT'],
#         tasks=['return_1step', 'return_4step'],
#         model_types=['lightgbm']
#     )
    
#     # Make predictions
#     predictions = predictor.predict_all_tasks('BTCUSDT')
#     return predictions

# def production_deployment_example():
#     """Full production deployment example"""
    
#     # Database configuration
#     db_config = {
#         "dbname": "crypto_trading",
#         "user": "your_user",
#         "password": "your_password",
#         "host": "localhost",
#         "port": "5432"
#     }
    
#     redis_config = {
#         'host': 'localhost',
#         'port': 6379,
#         'db': 0
#     }
    
#     # 1. Initialize services
#     trainer = ModelTrainingPipeline(db_config)
#     predictor = MultiTaskModelPredictor(db_config, redis_config)
    
#     # 2. Load production models
#     loaded, failed = predictor.load_production_models(
#         symbols=['BTCUSDT', 'ETHUSDT'],
#         tasks=['return_4step', 'direction_4step', 'volatility_4step']
#     )
    
#     print(f"Loaded {loaded} production models for inference")
    
#     # 3. Test predictions
#     test_predictions = predictor.predict_all_tasks('BTCUSDT')
    
#     # 4. Start API service (would be done separately)
#     print("Ready to start FastAPI service for production inference")
    
#     return test_predictions

# def model_lifecycle_management():
#     """Complete model lifecycle management"""
    
#     registry = MLflowModelRegistry()
    
#     # 1. Training → Staging promotion (automatic)
#     # This happens after training via promote_best_models_to_staging()
    
#     # 2. Staging → Production promotion (manual with criteria)
#     model_name = "crypto_lightgbm_return_4step_BTCUSDT"
    
#     # Get staging model performance
#     staging_version = registry.client.get_latest_versions(model_name, stages=["Staging"])[0]
    
#     # Manual review and promotion
#     registry.promote_model_to_production(model_name, staging_version.version)
    
#     # 3. Production → Archived (when replaced)
#     old_prod_version = "1"
#     registry.archive_model_version(model_name, old_prod_version)
    
#     print(f"Complete lifecycle: Training → Staging → Production → Archived")

# # Docker deployment configuration
# DOCKER_COMPOSE_CONFIG = """
# # docker-compose.yml for production deployment
# version: '3.8'
# services:
#   prediction-api:
#     build: .
#     ports:
#       - "8000:8000"
#     environment:
#       - DB_HOST=postgres
#       - DB_NAME=crypto_trading
#       - REDIS_HOST=redis
#       - MLFLOW_TRACKING_URI=http://mlflow:5000
#     depends_on:
#       - postgres
#       - redis
#       - mlflow
#     restart: unless-stopped
    
#   postgres:
#     image: postgres:13
#     environment:
#       - POSTGRES_DB=crypto_trading
#       - POSTGRES_USER=crypto_user
#       - POSTGRES_PASSWORD=your_password
#     volumes:
#       - postgres_data:/var/lib/postgresql/data
    
#   redis:
#     image: redis:6-alpine
#     volumes:
#       - redis_data:/data
    
#   mlflow:
#     image: python:3.9
#     command: mlflow server --host 0.0.0.0 --port 5000
#     ports:
#       - "5000:5000"
    
# volumes:
#   postgres_data:
#   redis_data:
# """

# if __name__ == "__main__":
#     main()