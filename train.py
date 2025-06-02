from transformers import Trainer, EarlyStoppingCallback
import time

class ProductionCRFTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.current_epoch = 0
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        if return_outputs:
            return loss, outputs
        return loss
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels", None)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]
            predictions = model.decode(logits, inputs.get("attention_mask"))
            batch_size, seq_len = logits.shape[:2]
            pred_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)
            for i, pred_seq in enumerate(predictions):
                pred_len = min(len(pred_seq), seq_len)
                pred_tensor[i, :pred_len] = torch.tensor(pred_seq[:pred_len])
            loss = None
            if labels is not None:
                outputs_with_labels = model(**inputs, labels=labels)
                loss = outputs_with_labels["loss"]
        return (loss, pred_tensor, labels)

# Add MonitoredTrainer and TrainingLogger classes from your notebook as well.
# (Copy their full code here.)

class TrainingLogger:
    """Comprehensive training logger with real-time monitoring"""
    
    def __init__(self, total_steps, total_examples, device):
        self.total_steps = total_steps
        self.total_examples = total_examples
        self.device = device
        self.step_times = []
        self.losses = []
        self.eval_metrics = []
        self.start_time = None
        self.current_step = 0
        self.best_f1 = 0.0
        
    def start_training(self):
        """Initialize training monitoring"""
        self.start_time = time.time()
        print("🚀 STARTING PRODUCTION-GRADE NER TRAINING WITH LIVE MONITORING")
        print("="*80)
        print("🎯 Target: 95%+ F1 Score with EMAIL/PHONE Detection")
        print(f"📊 Dataset: {self.total_examples:,} total examples")
        print("📈 Implementing Senior Staff Engineer recommendations")
        print(f"⏱️  Estimated total steps: {self.total_steps:,}")
        print("="*80)
        
        # Log initial system state
        if torch.cuda.is_available() and self.device.type == 'cuda':
            print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")
            print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB reserved")
        else:
            print(f"🖥️  Device: {self.device} (CPU mode)")
            
    def log_step(self, step, loss, learning_rate=None):
        """Log training step with real-time updates"""
        current_time = time.time()
        step_time = current_time - (self.start_time if len(self.step_times) == 0 else self.step_times[-1])
        self.step_times.append(current_time)
        self.losses.append(loss)
        self.current_step = step
        
        # Calculate progress and ETA
        progress = step / self.total_steps
        elapsed_time = current_time - self.start_time
        avg_step_time = elapsed_time / step if step > 0 else 0
        eta_seconds = avg_step_time * (self.total_steps - step)
        eta = timedelta(seconds=int(eta_seconds))
        
        # Log every 25 steps with detailed info
        if step % 25 == 0:
            print(f"Step {step:>6}/{self.total_steps} ({progress:>6.1%}) | "
                  f"Loss: {loss:>7.4f} | "
                  f"ETA: {str(eta):>8} | "
                  f"Step Time: {step_time:>5.2f}s")
            
            # Memory monitoring for GPU
            if torch.cuda.is_available() and self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_cached = torch.cuda.memory_reserved() / 1e9
                if step % 100 == 0:  # Less frequent memory logging
                    print(f"         Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")
    
    def log_epoch_end(self, epoch, eval_results=None):
        """Log end of epoch with comprehensive metrics"""
        elapsed = time.time() - self.start_time
        print("\n" + "="*60)
        print(f"📊 EPOCH {epoch + 1} COMPLETED")
        print(f"⏱️  Elapsed: {timedelta(seconds=int(elapsed))}")
        
        if eval_results:
            current_f1 = eval_results.get('eval_f1', 0)
            print(f"🎯 Validation F1: {current_f1:.4f}")
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                print(f"🏆 NEW BEST F1: {current_f1:.4f} (↑{current_f1 - self.best_f1:.4f})")
            
            # Log per-entity metrics if available
            for entity in ['per', 'org', 'email', 'phone']:
                f1_key = f'eval_{entity}_f1'
                if f1_key in eval_results:
                    f1_score = eval_results[f1_key]
                    status = "🟢" if f1_score >= 0.85 else "🟡" if f1_score >= 0.75 else "🔴"
                    print(f"   {entity.upper():<6}: {f1_score:.4f} {status}")
            
            self.eval_metrics.append({
                'epoch': epoch + 1,
                'f1': current_f1,
                'step': self.current_step,
                'time': elapsed
            })
        
        print("="*60 + "\n")
    
    def plot_progress(self):
        """Create live training progress plots"""
        if len(self.losses) < 10:  # Wait for some data
            return
            
        try:
            clear_output(wait=True)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curve
            steps = list(range(1, len(self.losses) + 1))
            ax1.plot(steps, self.losses, 'b-', alpha=0.7)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            
            # Loss moving average
            if len(self.losses) > 20:
                window = min(50, len(self.losses) // 4)
                moving_avg = pd.Series(self.losses).rolling(window=window).mean()
                ax1.plot(steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
                ax1.legend()
            
            # Step timing
            if len(self.step_times) > 1:
                step_durations = [self.step_times[i] - self.step_times[i-1] 
                                for i in range(1, len(self.step_times))]
                ax2.plot(range(1, len(step_durations) + 1), step_durations, 'g-', alpha=0.7)
                ax2.set_title('Step Duration')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Seconds')
                ax2.grid(True, alpha=0.3)
            
            # Validation F1 progress
            if self.eval_metrics:
                eval_steps = [m['step'] for m in self.eval_metrics]
                eval_f1s = [m['f1'] for m in self.eval_metrics]
                ax3.plot(eval_steps, eval_f1s, 'ro-', linewidth=2, markersize=6)
                ax3.set_title('Validation F1 Score')
                ax3.set_xlabel('Step')
                ax3.set_ylabel('F1 Score')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 1)
            
            # Progress bar simulation
            progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
            ax4.barh([0], [progress], color='green', alpha=0.7)
            ax4.barh([0], [1-progress], left=[progress], color='lightgray', alpha=0.7)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(-0.5, 0.5)
            ax4.set_title(f'Overall Progress: {progress:.1%}')
            ax4.set_xlabel('Completion')
            ax4.set_yticks([])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Plot update failed: {e}")


class MonitoredTrainer(ProductionCRFTrainer):
    """Enhanced trainer with comprehensive logging"""
    
    def __init__(self, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.plot_frequency = 100  # Plot every N steps
        
    def log(self, logs, *args, **kwargs):
        """Enhanced logging with real-time monitoring"""
        super().log(logs, *args, **kwargs)
        
        # Extract current step and loss
        current_step = self.state.global_step
        current_loss = logs.get('train_loss', logs.get('loss', 0))
        learning_rate = logs.get('learning_rate', None)
        
        # Log step information
        self.logger.log_step(current_step, current_loss, learning_rate)
        
        # Update plots periodically
        if current_step % self.plot_frequency == 0:
            self.logger.plot_progress()
        
        # Handle evaluation results
        if 'eval_f1' in logs:
            self.logger.log_epoch_end(self.state.epoch, logs)
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch"""
        # This ensures we capture epoch-end information
        super().on_epoch_end(args, state, control, model, **kwargs)