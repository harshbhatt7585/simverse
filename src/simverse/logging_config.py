"""Central logging configuration for Simverse with beautiful terminal output."""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

_DEFAULT_LEVEL = logging.INFO
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_FILE_ENV = "SIMVERSE_LOG_FILE"


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI Color Codes for Beautiful Terminal Output
# ═══════════════════════════════════════════════════════════════════════════════
class Colors:
    # Basic colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


# Box drawing characters for pretty borders
class Box:
    H = "─"  # horizontal
    V = "│"  # vertical
    TL = "┌"  # top-left
    TR = "┐"  # top-right
    BL = "└"  # bottom-left
    BR = "┘"  # bottom-right
    T = "┬"  # top tee
    B = "┴"  # bottom tee
    L = "├"  # left tee
    R = "┤"  # right tee
    X = "┼"  # cross
    
    # Double lines
    DH = "═"
    DV = "║"
    DTL = "╔"
    DTR = "╗"
    DBL = "╚"
    DBR = "╝"


def _ensure_log_dir(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = _DEFAULT_LEVEL, log_file: Optional[str] = None) -> None:
    """Configure root logger with console + optional file handlers."""
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)

    log_file = log_file or os.environ.get(_LOG_FILE_ENV)
    if not log_file:
        return

    log_path = Path(log_file).expanduser().resolve()
    _ensure_log_dir(log_path)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper to fetch module loggers."""
    return logging.getLogger(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Beautiful Training Logger
# ═══════════════════════════════════════════════════════════════════════════════
class TrainingLogger:
    """Beautiful terminal logger for training progress."""
    
    def __init__(self, name: str = "SimVerse", width: int = 80):
        self.name = name
        self.width = width
        self.start_time = None
        self.episode_start_time = None
        self.current_episode = 0
        self.total_episodes = 0
        
    def _print(self, text: str, end: str = "\n"):
        """Print with immediate flush."""
        print(text, end=end, flush=True)
    
    def _colorize(self, text: str, color: str) -> str:
        """Wrap text with color codes."""
        return f"{color}{text}{Colors.RESET}"
    
    def _center(self, text: str, width: int = None) -> str:
        """Center text within width."""
        width = width or self.width
        return text.center(width)
    
    def _progress_bar(self, current: int, total: int, width: int = 40, 
                      fill_char: str = "█", empty_char: str = "░") -> str:
        """Create a progress bar."""
        if total == 0:
            return empty_char * width
        progress = current / total
        filled = int(width * progress)
        bar = fill_char * filled + empty_char * (width - filled)
        percentage = progress * 100
        return f"{bar} {percentage:5.1f}%"
    
    def header(self, title: str = None):
        """Print a beautiful header."""
        title = title or self.name
        self._print("")
        self._print(f"{Colors.BRIGHT_CYAN}{Box.DTL}{Box.DH * (self.width - 2)}{Box.DTR}{Colors.RESET}")
        self._print(f"{Colors.BRIGHT_CYAN}{Box.DV}{Colors.RESET}{Colors.BOLD}{Colors.BRIGHT_WHITE}{self._center(f'🌾 {title} 🌾')}{Colors.RESET}{Colors.BRIGHT_CYAN}{Box.DV}{Colors.RESET}")
        self._print(f"{Colors.BRIGHT_CYAN}{Box.DBL}{Box.DH * (self.width - 2)}{Box.DBR}{Colors.RESET}")
        self._print("")
    
    def section(self, title: str):
        """Print a section header."""
        self._print(f"\n{Colors.BRIGHT_BLUE}{'─' * 3} {title} {'─' * (self.width - len(title) - 5)}{Colors.RESET}")
    
    def config(self, config: Dict[str, Any]):
        """Print configuration in a nice table."""
        self.section("⚙️  Configuration")
        max_key_len = max(len(str(k)) for k in config.keys())
        for key, value in config.items():
            key_str = f"{Colors.CYAN}{key:<{max_key_len}}{Colors.RESET}"
            value_str = f"{Colors.BRIGHT_WHITE}{value}{Colors.RESET}"
            self._print(f"  {key_str} : {value_str}")
    
    def start_training(self, total_episodes: int):
        """Mark training start."""
        self.start_time = time.time()
        self.total_episodes = total_episodes
        self.section("🚀 Training Started")
        self._print(f"  {Colors.DIM}Total episodes: {total_episodes}{Colors.RESET}")
        self._print("")
    
    def start_episode(self, episode: int):
        """Mark episode start."""
        self.current_episode = episode
        self.episode_start_time = time.time()
    
    def log_step(self, step: int, total_steps: int, metrics: Dict[str, float] = None):
        """Log a training step with inline update."""
        progress = self._progress_bar(step, total_steps, width=30)
        
        metrics_str = ""
        if metrics:
            parts = []
            for k, v in metrics.items():
                if "loss" in k.lower():
                    color = Colors.YELLOW
                elif "reward" in k.lower():
                    color = Colors.GREEN if v > 0 else Colors.RED
                else:
                    color = Colors.WHITE
                parts.append(f"{color}{k}={v:.4f}{Colors.RESET}")
            metrics_str = " │ " + " ".join(parts)
        
        episode_str = f"{Colors.BRIGHT_MAGENTA}Ep {self.current_episode:3d}/{self.total_episodes}{Colors.RESET}"
        step_str = f"{Colors.DIM}Step {step:4d}/{total_steps}{Colors.RESET}"
        
        self._print(f"\r  {episode_str} │ {step_str} │ {progress}{metrics_str}", end="")
    
    def log_epoch(self, epoch: int, total_epochs: int, policy_loss: float, value_loss: float):
        """Log epoch results."""
        # Color code losses
        policy_color = Colors.BRIGHT_YELLOW if policy_loss > 0.1 else Colors.GREEN
        value_color = Colors.BRIGHT_YELLOW if value_loss > 0.1 else Colors.GREEN
        
        epoch_str = f"{Colors.CYAN}Epoch {epoch + 1:2d}/{total_epochs}{Colors.RESET}"
        policy_str = f"{policy_color}π_loss={policy_loss:.4f}{Colors.RESET}"
        value_str = f"{value_color}v_loss={value_loss:.4f}{Colors.RESET}"
        
        self._print(f"    {epoch_str} │ {policy_str} │ {value_str}")
    
    def end_episode(self, episode: int, total_reward: float, avg_reward: float, steps: int):
        """Mark episode end with summary."""
        elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        self._print("")  # Clear the step line
        
        # Episode summary box
        reward_color = Colors.BRIGHT_GREEN if total_reward > 0 else Colors.BRIGHT_RED
        
        self._print(f"  {Colors.GREEN}✓{Colors.RESET} Episode {episode} complete in {elapsed:.1f}s")
        self._print(f"    {Colors.DIM}├─{Colors.RESET} Total Reward: {reward_color}{total_reward:+.2f}{Colors.RESET}")
        self._print(f"    {Colors.DIM}├─{Colors.RESET} Avg Reward:   {Colors.WHITE}{avg_reward:.4f}{Colors.RESET}")
        self._print(f"    {Colors.DIM}└─{Colors.RESET} Steps:        {Colors.WHITE}{steps}{Colors.RESET}")
        self._print("")
    
    def metric(self, name: str, value: float, unit: str = ""):
        """Log a single metric."""
        self._print(f"  {Colors.CYAN}{name}:{Colors.RESET} {Colors.BRIGHT_WHITE}{value:.4f}{Colors.RESET} {Colors.DIM}{unit}{Colors.RESET}")
    
    def success(self, message: str):
        """Print success message."""
        self._print(f"  {Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")
    
    def warning(self, message: str):
        """Print warning message."""
        self._print(f"  {Colors.BRIGHT_YELLOW}⚠ {message}{Colors.RESET}")
    
    def error(self, message: str):
        """Print error message."""
        self._print(f"  {Colors.BRIGHT_RED}✗ {message}{Colors.RESET}")
    
    def info(self, message: str):
        """Print info message."""
        self._print(f"  {Colors.BRIGHT_BLUE}ℹ {message}{Colors.RESET}")
    
    def finish(self, final_stats: Dict[str, float] = None):
        """Print training complete summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        self._print("")
        self._print(f"{Colors.BRIGHT_GREEN}{Box.DTL}{Box.DH * (self.width - 2)}{Box.DTR}{Colors.RESET}")
        self._print(f"{Colors.BRIGHT_GREEN}{Box.DV}{Colors.RESET}{Colors.BOLD}{Colors.BRIGHT_WHITE}{self._center('🎉 Training Complete! 🎉')}{Colors.RESET}{Colors.BRIGHT_GREEN}{Box.DV}{Colors.RESET}")
        self._print(f"{Colors.BRIGHT_GREEN}{Box.DBL}{Box.DH * (self.width - 2)}{Box.DBR}{Colors.RESET}")
        
        # Time stats
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        self._print(f"\n  {Colors.DIM}Total time:{Colors.RESET} {Colors.BRIGHT_WHITE}{time_str}{Colors.RESET}")
        self._print(f"  {Colors.DIM}Episodes:{Colors.RESET}   {Colors.BRIGHT_WHITE}{self.total_episodes}{Colors.RESET}")
        
        if final_stats:
            self._print(f"\n  {Colors.UNDERLINE}Final Metrics:{Colors.RESET}")
            for key, value in final_stats.items():
                self._print(f"    {Colors.CYAN}{key}:{Colors.RESET} {Colors.BRIGHT_WHITE}{value:.4f}{Colors.RESET}")
        
        self._print("")


# Global training logger instance
training_logger = TrainingLogger()
