import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

class StockAnalyzer:


def __init__(self, token=None):
    """
    初始化分析器
    token: Tushare API token，如果没有请先注册获取
    """
    if token:
        ts.set_token(token)
    self.pro = ts.pro_api()
    
def get_market_data(self, start_date=None, end_date=None):
    """获取市场基础数据"""
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
    print(f"正在获取 {start_date} 到 {end_date} 的市场数据...")
    
    # 获取主要指数数据
    indices = {
        '000001.SH': '上证指数',
        '399001.SZ': '深证成指',
        '399006.SZ': '创业板指'
    }
    
    self.index_data = {}
    for code, name in indices.items():
        try:
            data = self.pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
            data['trade_date'] = pd.to_datetime(data['trade_date'])
            data = data.sort_values('trade_date')
            self.index_data[name] = data
            print(f"✓ 成功获取{name}数据: {len(data)}条记录")
        except Exception as e:
            print(f"✗ 获取{name}数据失败: {e}")
            
    # 获取热门股票数据
    self.get_popular_stocks(start_date, end_date)
    
def get_popular_stocks(self, start_date, end_date):
    """获取热门股票数据"""
    # 选择一些知名股票
    popular_stocks = {
        '000001.SZ': '平安银行',
        '000002.SZ': '万科A',
        '600036.SH': '招商银行',
        '600519.SH': '贵州茅台',
        '000858.SZ': '五粮液'
    }
    
    self.stock_data = {}
    for code, name in popular_stocks.items():
        try:
            data = self.pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            data['trade_date'] = pd.to_datetime(data['trade_date'])
            data = data.sort_values('trade_date')
            data['returns'] = data['close'].pct_change()
            self.stock_data[name] = data
            print(f"✓ 成功获取{name}数据: {len(data)}条记录")
        except Exception as e:
            print(f"✗ 获取{name}数据失败: {e}")
            
def calculate_technical_indicators(self):
        """计算技术指标"""
        print("\n正在计算技术指标...")
        
        for name, data in self.index_data.items():
            # 移动平均线
            data['MA5'] = data['close'].rolling(window=5).mean()
            data['MA20'] = data['close'].rolling(window=20).mean()
            data['MA60'] = data['close'].rolling(window=60).mean()
            
            # RSI指标
            data['RSI'] = self.calculate_rsi(data['close'])
            
            # 布林带
            data['BB_upper'], data['BB_lower'] = self.calculate_bollinger_bands(data['close'])
        
        for name, data in self.stock_data.items():
            # 移动平均线
            data['MA5'] = data['close'].rolling(window=5).mean()
            data['MA20'] = data['close'].rolling(window=20).mean()
            data['MA60'] = data['close'].rolling(window=60).mean()
            
            # RSI指标
            data['RSI'] = self.calculate_rsi(data['close'])
            
            # 布林带
            data['BB_upper'], data['BB_lower'] = self.calculate_bollinger_bands(data['close'])
        
        print("✓ 技术指标计算完成")
        
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
    """计算布林带"""
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    return upper_band, lower_band
    
def analyze_market_performance(self):
    """分析市场表现"""
    print("\n正在分析市场表现...")
    
    self.market_analysis = {}
    
    # 分析指数表现
    for name, data in self.index_data.items():
        analysis = {
            '当前价格': data['close'].iloc[-1],
            '期初价格': data['close'].iloc[0],
            '总收益率': ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100,
            '日均收益率': data['close'].pct_change().mean() * 100,
            '波动率': data['close'].pct_change().std() * 100 * np.sqrt(252),
            '最大回撤': self.calculate_max_drawdown(data['close']),
            '夏普比率': self.calculate_sharpe_ratio(data['close'].pct_change())
        }
        self.market_analysis[name] = analysis
        
    # 分析个股表现
    for name, data in self.stock_data.items():
        analysis = {
            '当前价格': data['close'].iloc[-1],
            '期初价格': data['close'].iloc[0],
            '总收益率': ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100,
            '日均收益率': data['close'].pct_change().mean() * 100,
            '波动率': data['close'].pct_change().std() * 100 * np.sqrt(252),
            '最大回撤': self.calculate_max_drawdown(data['close']),
            '夏普比率': self.calculate_sharpe_ratio(data['close'].pct_change())
        }
        self.market_analysis[name] = analysis
        
    print("✓ 市场表现分析完成")
    
def calculate_max_drawdown(self, prices):
    """计算最大回撤"""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() * 100
    
def calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
    """计算夏普比率"""
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    return excess_returns / volatility if volatility != 0 else 0
    
def create_visualizations(self):
    """创建可视化图表"""
    print("\n正在创建可视化图表...")
    
    # 设置图表布局
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 主要指数走势图
    plt.subplot(3, 3, 1)
    for name, data in self.index_data.items():
        normalized_price = data['close'] / data['close'].iloc[0] * 100
        plt.plot(data['trade_date'], normalized_price, label=name, linewidth=2)
    plt.title('主要指数走势对比（标准化）', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('标准化价格（基期=100）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 个股价格走势
    plt.subplot(3, 3, 2)
    stock_names = list(self.stock_data.keys())[:3]  # 选择前3只股票
    for name in stock_names:
        data = self.stock_data[name]
        plt.plot(data['trade_date'], data['close'], label=name, linewidth=2)
    plt.title('热门个股价格走势', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('股价（元）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 收益率分布
    plt.subplot(3, 3, 3)
    all_returns = []
    labels = []
    for name, data in list(self.index_data.items())[:2]:
        returns = data['close'].pct_change().dropna() * 100
        all_returns.append(returns)
        labels.append(name)
    plt.boxplot(all_returns, labels=labels)
    plt.title('指数日收益率分布', fontsize=14, fontweight='bold')
    plt.ylabel('日收益率（%）')
    plt.grid(True, alpha=0.3)
    
    # 4. 技术指标分析（以上证指数为例）
    plt.subplot(3, 3, 4)
    if '上证指数' in self.index_data:
        data = self.index_data['上证指数']
        plt.plot(data['trade_date'], data['close'], label='收盘价', linewidth=2)
        plt.plot(data['trade_date'], data['MA5'], label='MA5', alpha=0.8)
        plt.plot(data['trade_date'], data['MA20'], label='MA20', alpha=0.8)
        plt.fill_between(data['trade_date'], data['BB_upper'], data['BB_lower'], 
                       alpha=0.2, label='布林带')
    plt.title('上证指数技术分析', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. RSI指标
    plt.subplot(3, 3, 5)
    if '上证指数' in self.index_data:
        data = self.index_data['上证指数']
        plt.plot(data['trade_date'], data['RSI'], color='purple', linewidth=2)
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线')
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线')
    plt.title('上证指数RSI指标', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('RSI')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 成交量分析
    plt.subplot(3, 3, 6)
    if '上证指数' in self.index_data:
        data = self.index_data['上证指数']
        plt.bar(data['trade_date'], data['vol'], alpha=0.6, color='orange')
        plt.plot(data['trade_date'], data['vol'].rolling(20).mean(), 
                color='red', linewidth=2, label='20日均量')
    plt.title('上证指数成交量', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('成交量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. 市场表现热力图
    plt.subplot(3, 3, 7)
    performance_data = []
    names = []
    for name, analysis in self.market_analysis.items():
        performance_data.append([
            analysis['总收益率'],
            analysis['波动率'],
            analysis['夏普比率']
        ])
        names.append(name)
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data, 
                                    columns=['总收益率', '波动率', '夏普比率'],
                                    index=names)
        sns.heatmap(performance_df, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
    plt.title('市场表现热力图', fontsize=14, fontweight='bold')
    
    # 8. 风险收益散点图
    plt.subplot(3, 3, 8)
    returns = []
    risks = []
    names_list = []
    for name, analysis in self.market_analysis.items():
        returns.append(analysis['总收益率'])
        risks.append(analysis['波动率'])
        names_list.append(name)
    
    if returns:
        plt.scatter(risks, returns, s=100, alpha=0.7, c=range(len(returns)), cmap='viridis')
        for i, name in enumerate(names_list):
            plt.annotate(name, (risks[i], returns[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
    plt.title('风险-收益分析', fontsize=14, fontweight='bold')
    plt.xlabel('年化波动率（%）')
    plt.ylabel('总收益率（%）')
    plt.grid(True, alpha=0.3)
    
    # 9. 相关性分析
    plt.subplot(3, 3, 9)
    if len(self.index_data) >= 2:
        correlation_data = pd.DataFrame()
        for name, data in self.index_data.items():
            correlation_data[name] = data['close'].pct_change()
        corr_matrix = correlation_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('指数相关性分析', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('market_analysis_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 可视化图表创建完成，已保存为 'market_analysis_report.png'")
    
def generate_report(self):
    """生成分析报告"""
    print("\n" + "="*60)
    print("           中国A股市场分析报告")
    print("="*60)
    
    print(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n【市场概况】")
    print("-" * 40)
    
    for name, analysis in self.market_analysis.items():
        print(f"\n{name}:")
        print(f"  当前价格: {analysis['当前价格']:.2f}")
        print(f"  总收益率: {analysis['总收益率']:.2f}%")
        print(f"  年化波动率: {analysis['波动率']:.2f}%")
        print(f"  最大回撤: {analysis['最大回撤']:.2f}%")
        print(f"  夏普比率: {analysis['夏普比率']:.3f}")
        
    print("\n【投资建议】")
    print("-" * 40)
    
    # 基于分析结果给出建议
    best_performer = max(self.market_analysis.items(), 
                       key=lambda x: x[1]['总收益率'])
    lowest_risk = min(self.market_analysis.items(),
                     key=lambda x: x[1]['波动率'])
    best_sharpe = max(self.market_analysis.items(),
                     key=lambda x: x[1]['夏普比率'])
    
    print(f"• 收益表现最佳: {best_performer[0]} ({best_performer[1]['总收益率']:.2f}%)")
    print(f"• 风险最低标的: {lowest_risk[0]} (波动率{lowest_risk[1]['波动率']:.2f}%)")
    print(f"• 风险调整收益最优: {best_sharpe[0]} (夏普比率{best_sharpe[1]['夏普比率']:.3f})")
    
    print("\n【风险提示】")
    print("-" * 40)
    print("• 投资有风险，入市需谨慎")
    print("• 本报告仅供参考，不构成投资建议")
    print("• 请根据个人风险承受能力做出投资决策")
    
    print("\n" + "="*60)
    
def run_analysis(self, token=None, days_back=180):
        """运行完整分析流程"""
        print("开始中国A股市场数据分析...")
        print("="*50)
        
        # 如果没有提供token，提供提示信息并使用模拟数据
        if not token:
            print("⚠️  未提供Tushare API Token")
            print("请访问 https://tushare.pro 注册并获取免费API Token")
            print("然后使用: analyzer.run_analysis(token='your_token_here')")
            print("\n使用模拟数据进行演示...")
            self.create_sample_data()
            # ---- 修复：对模拟数据也要计算技术指标 ----
            self.calculate_technical_indicators()
        else:
            # 设置日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            
            # 执行分析流程
            self.get_market_data(start_date, end_date)
            self.calculate_technical_indicators()
            
        self.analyze_market_performance()
        self.create_visualizations()
        self.generate_report()
        
        print("\n✅ 分析完成！图表已保存为 'market_analysis_report.png'")
    
def create_sample_data(self):
    """创建模拟数据用于演示"""
    print("正在创建模拟数据...")
    
    # 生成模拟的指数数据
    dates = pd.date_range(start='2023-06-01', end='2024-06-01', freq='D')
    dates = dates[dates.dayofweek < 5]  # 只保留工作日
    
    np.random.seed(42)
    
    # 模拟上证指数
    base_price = 3200
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
        
    shanghai_data = pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'vol': np.random.lognormal(15, 0.5, len(dates))
    })
    
    # 模拟深证成指
    base_price = 12000
    returns = np.random.normal(0.0003, 0.018, len(dates))
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
        
    shenzhen_data = pd.DataFrame({
        'trade_date': dates,
        'close': prices,
        'vol': np.random.lognormal(14.8, 0.4, len(dates))
    })
    
    self.index_data = {
        '上证指数': shanghai_data,
        '深证成指': shenzhen_data
    }
    
    # 模拟个股数据
    stock_configs = {
        '贵州茅台': {'base': 1800, 'vol': 0.025},
        '招商银行': {'base': 35, 'vol': 0.022},
        '平安银行': {'base': 12, 'vol': 0.028}
    }
    
    self.stock_data = {}
    for name, config in stock_configs.items():
        returns = np.random.normal(0.0008, config['vol'], len(dates))
        prices = [config['base']]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
            
        stock_df = pd.DataFrame({
            'trade_date': dates,
            'close': prices,
            'vol': np.random.lognormal(12, 0.6, len(dates)),
            'returns': returns
        })
        self.stock_data[name] = stock_df
        
    print("✓ 模拟数据创建完成")


# 使用示例

if __name__ == "__main__":
# 创建分析器实例
analyzer = StockAnalyzer()


# 运行分析（不提供token时将使用模拟数据）
analyzer.run_analysis(token='a394376ad4ede9c1214c06479e7d1bee32919045f909fcbfc39cabde', days_back=365)

# 如果您有Tushare API token，请使用以下方式：
# analyzer.run_analysis(token='your_tushare_token_here', days_back=365)
