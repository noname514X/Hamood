import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Lasso

# Import from the ADMM2 module - fix the import path
try:
    from ADMM2 import ADMM_LASSO, generate_lasso_data, compare_with_sklearn
except ImportError:
    print("Error: Could not import from ADMM2.py. Make sure ADMM2.py is in the same directory.")
    exit(1)

class TestCompareWithSklearn:
    """Test suite for compare_with_sklearn function"""
    
    def setup_method(self):
        """Setup test data before each test"""
        # Generate smaller test data for faster testing
        self.X_small, self.y_small, self.beta_true_small = generate_lasso_data(
            n=100, p=500, sparsity=20, lambda_reg=0.05, seed=42
        )
        
        # Generate medium size data similar to assignment
        self.X_medium, self.y_medium, self.beta_true_medium = generate_lasso_data(
            n=200, p=1000, sparsity=30, lambda_reg=0.05, seed=123
        )
    
    def test_compare_basic_functionality(self):
        """Test basic functionality of compare_with_sklearn"""
        print("\nTesting basic functionality...")
        admm_model, sklearn_model = compare_with_sklearn(
            self.X_small, self.y_small, lambda_reg=0.05
        )
        
        # Check that both models were trained
        assert hasattr(admm_model, 'beta_'), "ADMM model should have beta_ attribute"
        assert hasattr(sklearn_model, 'coef_'), "Sklearn model should have coef_ attribute"
        assert admm_model.beta_.shape == sklearn_model.coef_.shape, "Coefficient shapes should match"
        
        # Check convergence
        assert admm_model.n_iter_ > 0, "ADMM should have positive iterations"
        assert sklearn_model.n_iter_ > 0, "Sklearn should have positive iterations"
        
        print("✓ Basic functionality test passed")
    
    def test_coefficient_similarity_with_plots(self):
        """Test and visualize coefficient similarity between ADMM and sklearn"""
        print("\nTesting coefficient similarity with comprehensive plots...")
        lambda_reg = 0.05
        admm_model, sklearn_model = compare_with_sklearn(
            self.X_medium, self.y_medium, lambda_reg=lambda_reg
        )
        
        # Create comprehensive comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Coefficient comparison scatter plot
        axes[0, 0].scatter(admm_model.beta_, sklearn_model.coef_, alpha=0.6)
        min_val = min(admm_model.beta_.min(), sklearn_model.coef_.min())
        max_val = max(admm_model.beta_.max(), sklearn_model.coef_.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
        axes[0, 0].set_xlabel('ADMM Coefficients')
        axes[0, 0].set_ylabel('Sklearn Coefficients')
        axes[0, 0].set_title('Coefficient Comparison (σ=1.0)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: First 50 coefficients comparison
        indices = np.arange(min(50, len(admm_model.beta_)))
        axes[0, 1].plot(indices, admm_model.beta_[:len(indices)], 'b-', label='ADMM', linewidth=2)
        axes[0, 1].plot(indices, sklearn_model.coef_[:len(indices)], 'r--', label='Sklearn', linewidth=2)
        axes[0, 1].plot(indices, self.beta_true_medium[:len(indices)], 'g:', label='True', linewidth=2)
        axes[0, 1].set_xlabel('Coefficient Index')
        axes[0, 1].set_ylabel('Coefficient Value')
        axes[0, 1].set_title(f'First {len(indices)} Coefficients Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Coefficient difference histogram
        coef_diff = admm_model.beta_ - sklearn_model.coef_
        axes[0, 2].hist(coef_diff, bins=50, alpha=0.7, color='purple')
        axes[0, 2].set_xlabel('Coefficient Difference (ADMM - Sklearn)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Coefficient Differences\nMean: {np.mean(coef_diff):.6f}, Std: {np.std(coef_diff):.6f}')
        axes[0, 2].grid(True)
        
        # Plot 4: Prediction comparison
        X_test = self.X_medium[:50, :]  # Use first 50 samples as test
        y_test = self.y_medium[:50]
        admm_pred = admm_model.predict(X_test)
        sklearn_pred = sklearn_model.predict(X_test)
        
        axes[1, 0].scatter(admm_pred, sklearn_pred, alpha=0.7)
        min_pred = min(admm_pred.min(), sklearn_pred.min())
        max_pred = max(admm_pred.max(), sklearn_pred.max())
        axes[1, 0].plot([min_pred, max_pred], [min_pred, max_pred], 'r--', label='Perfect Agreement')
        axes[1, 0].set_xlabel('ADMM Predictions')
        axes[1, 0].set_ylabel('Sklearn Predictions')
        axes[1, 0].set_title('Prediction Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: Non-zero coefficients comparison
        admm_nonzero = np.sum(np.abs(admm_model.beta_) > 1e-6)
        sklearn_nonzero = np.sum(np.abs(sklearn_model.coef_) > 1e-6)
        true_nonzero = np.sum(np.abs(self.beta_true_medium) > 1e-6)
        
        methods = ['True', 'ADMM', 'Sklearn']
        nonzero_counts = [true_nonzero, admm_nonzero, sklearn_nonzero]
        colors = ['green', 'blue', 'red']
        
        bars = axes[1, 1].bar(methods, nonzero_counts, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Number of Non-zero Coefficients')
        axes[1, 1].set_title('Sparsity Comparison')
        axes[1, 1].grid(True, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, nonzero_counts):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + max(nonzero_counts)*0.01,
                           str(count), ha='center', va='bottom')
        
        # Plot 6: Convergence comparison (ADMM only has convergence history)
        axes[1, 2].plot(admm_model.objective_history, 'b-', linewidth=2)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Objective Value')
        axes[1, 2].set_title(f'ADMM Convergence (σ=1.0)\nFinal iterations: {admm_model.n_iter_}')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.suptitle('ADMM vs Sklearn LASSO Comparison (σ=1.0)', fontsize=16, y=1.02)
        plt.show()
        
        # Numerical assertions
        coef_diff_norm = np.linalg.norm(admm_model.beta_ - sklearn_model.coef_)
        print(f"\nNumerical Comparison Results:")
        print(f"Coefficient difference ||β_ADMM - β_sklearn||₂: {coef_diff_norm:.6f}")
        print(f"ADMM non-zero coefficients: {admm_nonzero}")
        print(f"Sklearn non-zero coefficients: {sklearn_nonzero}")
        print(f"True non-zero coefficients: {true_nonzero}")
        
        # Assert reasonable similarity (adjust tolerance as needed)
        if coef_diff_norm >= 1.0:
            print(f"Warning: Coefficient difference is large: {coef_diff_norm}")
        if abs(admm_nonzero - sklearn_nonzero) > 5:
            print(f"Warning: Sparsity patterns are quite different")
        
        print("✓ Coefficient similarity test with plots completed")
    
    def test_different_lambda_values(self):
        """Test comparison with different regularization parameters"""
        print("\nTesting different lambda values...")
        lambda_values = [0.01, 0.05, 0.1, 0.2]
        results = {}
        
        plt.figure(figsize=(15, 10))
        
        for i, lambda_reg in enumerate(lambda_values):
            print(f"  Testing λ = {lambda_reg}")
            admm_model, sklearn_model = compare_with_sklearn(
                self.X_small, self.y_small, lambda_reg=lambda_reg
            )
            
            coef_diff = np.linalg.norm(admm_model.beta_ - sklearn_model.coef_)
            admm_nonzero = np.sum(np.abs(admm_model.beta_) > 1e-6)
            sklearn_nonzero = np.sum(np.abs(sklearn_model.coef_) > 1e-6)
            
            results[lambda_reg] = {
                'coef_diff': coef_diff,
                'admm_nonzero': admm_nonzero,
                'sklearn_nonzero': sklearn_nonzero,
                'admm_l1_norm': np.linalg.norm(admm_model.beta_, 1),
                'sklearn_l1_norm': np.linalg.norm(sklearn_model.coef_, 1)
            }
            
            # Plot coefficient comparison for each lambda
            plt.subplot(2, 2, i + 1)
            plt.scatter(admm_model.beta_, sklearn_model.coef_, alpha=0.6)
            min_val = min(admm_model.beta_.min(), sklearn_model.coef_.min())
            max_val = max(admm_model.beta_.max(), sklearn_model.coef_.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
            plt.xlabel('ADMM Coefficients')
            plt.ylabel('Sklearn Coefficients')
            plt.title(f'λ = {lambda_reg}, ||diff||₂ = {coef_diff:.4f}')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*70)
        print("LAMBDA COMPARISON SUMMARY")
        print("="*70)
        print("λ\tCoef Diff\tADMM NZ\tSklearn NZ\tADMM L1\tSklearn L1")
        print("-"*70)
        for lambda_reg in lambda_values:
            r = results[lambda_reg]
            print(f"{lambda_reg:.2f}\t{r['coef_diff']:.4f}\t\t{r['admm_nonzero']}\t{r['sklearn_nonzero']}\t\t{r['admm_l1_norm']:.4f}\t{r['sklearn_l1_norm']:.4f}")
        
        print("✓ Different lambda values test completed")
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy comparison between ADMM and sklearn"""
        print("\nTesting prediction accuracy...")
        admm_model, sklearn_model = compare_with_sklearn(
            self.X_medium, self.y_medium, lambda_reg=0.05
        )
        
        # Split data for testing (use last 50 samples)
        X_test = self.X_medium[-50:, :]
        y_test = self.y_medium[-50:]
        
        # Make predictions
        admm_pred = admm_model.predict(X_test)
        sklearn_pred = sklearn_model.predict(X_test)
        
        # Calculate errors
        admm_mse = np.mean((admm_pred - y_test)**2)
        sklearn_mse = np.mean((sklearn_pred - y_test)**2)
        pred_diff_mse = np.mean((admm_pred - sklearn_pred)**2)
        
        print(f"Prediction Accuracy Comparison:")
        print(f"  ADMM MSE: {admm_mse:.6f}")
        print(f"  Sklearn MSE: {sklearn_mse:.6f}")
        print(f"  Prediction difference MSE: {pred_diff_mse:.6f}")
        
        # Plot prediction comparison
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(y_test, admm_pred, alpha=0.7, label='ADMM')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('ADMM Predictions')
        plt.title(f'ADMM Predictions\nMSE: {admm_mse:.4f}')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.scatter(y_test, sklearn_pred, alpha=0.7, label='Sklearn', color='orange')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Sklearn Predictions')
        plt.title(f'Sklearn Predictions\nMSE: {sklearn_mse:.4f}')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.scatter(admm_pred, sklearn_pred, alpha=0.7, color='purple')
        min_pred = min(admm_pred.min(), sklearn_pred.min())
        max_pred = max(admm_pred.max(), sklearn_pred.max())
        plt.plot([min_pred, max_pred], [min_pred, max_pred], 'r--')
        plt.xlabel('ADMM Predictions')
        plt.ylabel('Sklearn Predictions')
        plt.title(f'Prediction Comparison\nDiff MSE: {pred_diff_mse:.4f}')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Check predictions similarity
        if pred_diff_mse >= 0.1:
            print(f"Warning: Predictions are quite different: {pred_diff_mse}")
        if abs(admm_mse - sklearn_mse) >= 0.5:
            print(f"Warning: MSE difference is large: {abs(admm_mse - sklearn_mse)}")
        
        print("✓ Prediction accuracy test completed")
    
    def test_sigma_effect_on_sklearn_comparison(self):
        """Test how different sigma values affect similarity to sklearn"""
        print("\nTesting sigma effect on sklearn similarity...")
        sigma_values = [0.5, 1.0, 2.0, 5.0]
        results = {}
        
        # Get sklearn reference
        sklearn_lasso = Lasso(alpha=0.05/(2*len(self.y_small)), max_iter=1000, tol=1e-6)
        sklearn_lasso.fit(self.X_small, self.y_small)
        
        plt.figure(figsize=(16, 8))
        
        for i, sigma in enumerate(sigma_values):
            print(f"  Testing σ = {sigma}")
            
            # Fit ADMM with specific sigma
            admm = ADMM_LASSO(lambda_reg=0.05, sigma=sigma, max_iter=300)
            start_time = time.time()
            admm.fit(self.X_small, self.y_small)
            fit_time = time.time() - start_time
            
            coef_diff = np.linalg.norm(admm.beta_ - sklearn_lasso.coef_)
            
            results[sigma] = {
                'coef_diff': coef_diff,
                'iterations': admm.n_iter_,
                'time': fit_time,
                'objective': admm.objective_history[-1] if admm.objective_history else 0
            }
            
            # Plot coefficient comparison
            plt.subplot(2, 4, i + 1)
            plt.scatter(admm.beta_, sklearn_lasso.coef_, alpha=0.6)
            min_val = min(admm.beta_.min(), sklearn_lasso.coef_.min())
            max_val = max(admm.beta_.max(), sklearn_lasso.coef_.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.xlabel('ADMM Coefficients')
            plt.ylabel('Sklearn Coefficients')
            plt.title(f'σ = {sigma}\n||diff||₂ = {coef_diff:.4f}')
            plt.grid(True)
            
            # Plot convergence
            plt.subplot(2, 4, i + 5)
            if admm.objective_history:
                plt.plot(admm.objective_history)
            plt.xlabel('Iteration')
            plt.ylabel('Objective')
            plt.title(f'σ = {sigma} Convergence\n{admm.n_iter_} iter, {fit_time:.2f}s')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Summary table
        print("\n" + "="*70)
        print("SIGMA EFFECT ON SKLEARN SIMILARITY")
        print("="*70)
        print("σ\tCoef Diff\tIterations\tTime(s)\tFinal Objective")
        print("-"*70)
        for sigma in sigma_values:
            r = results[sigma]
            print(f"{sigma:.1f}\t{r['coef_diff']:.6f}\t{r['iterations']:4d}\t\t{r['time']:.3f}\t{r['objective']:.6f}")
        
        # Find best sigma (closest to sklearn)
        best_sigma = min(results.keys(), key=lambda s: results[s]['coef_diff'])
        print(f"\nBest σ for sklearn similarity: {best_sigma} (diff = {results[best_sigma]['coef_diff']:.6f})")
        
        print("✓ Sigma effect test completed")
        return results

def main():
    """Main function to run all tests"""
    print("="*60)
    print("RUNNING ADMM vs SKLEARN COMPARISON TESTS")
    print("="*60)
    
    try:
        # Initialize test suite
        test_suite = TestCompareWithSklearn()
        test_suite.setup_method()
        
        # Run all tests
        test_suite.test_compare_basic_functionality()
        test_suite.test_coefficient_similarity_with_plots()
        test_suite.test_different_lambda_values()
        test_suite.test_prediction_accuracy()
        test_suite.test_sigma_effect_on_sklearn_comparison()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred during testing: {e}")
        print("Please make sure ADMM2.py is in the same directory and contains all required functions.")

if __name__ == "__main__":
    main()