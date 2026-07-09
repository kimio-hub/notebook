---
title: Kimio
description: Kimio 的个人主页——计算机专业本科生，强化学习 / 大语言模型方向。自我介绍、学习笔记与联系方式。
---

<style>
  .home-card { transition: border-color 0.2s ease, transform 0.2s ease; }
  .home-card:hover { border-color: var(--secondary); transform: translateY(-2px); }
  .hero-btn { transition: opacity 0.2s ease, border-color 0.2s ease; }
  .hero-btn:hover { opacity: 0.82; border-color: var(--secondary); }
</style>

<div style="text-align:center;padding:2rem 0 1.4rem;">
  <img src="./avatar.jpg" alt="Kimio 的头像" width="128" height="128" style="border-radius:50%;object-fit:cover;box-shadow:0 12px 36px rgba(0,0,0,0.18);" />
  <h1 style="margin:1.1rem 0 0.25rem;font-size:2.1rem;letter-spacing:0.02em;">Kimio</h1>
  <p style="margin:0;color:var(--darkgray);font-size:1.02rem;">计算机专业本科生 · 强化学习 / 大语言模型</p>
  <p style="margin:0.55rem auto 0;max-width:34rem;color:var(--gray);font-size:0.93rem;line-height:1.6;">你好，欢迎来到我的主页。这里是我在互联网上的名片：<br/>关于我、我的学习笔记，以及找到我的方式。</p>
  <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:1rem;">
    <span style="border:1px solid var(--lightgray);border-radius:999px;padding:3px 13px;font-size:0.82rem;color:var(--darkgray);">强化学习</span>
    <span style="border:1px solid var(--lightgray);border-radius:999px;padding:3px 13px;font-size:0.82rem;color:var(--darkgray);">LLM 推理</span>
    <span style="border:1px solid var(--lightgray);border-radius:999px;padding:3px 13px;font-size:0.82rem;color:var(--darkgray);">多模态检索</span>
    <span style="border:1px solid var(--lightgray);border-radius:999px;padding:3px 13px;font-size:0.82rem;color:var(--darkgray);">算法博弈论</span>
  </div>
  <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:1.2rem;">
    <a class="hero-btn" href="https://github.com/kimio-hub" style="background:var(--secondary);color:var(--light);border:1px solid var(--secondary);border-radius:8px;padding:7px 18px;text-decoration:none;font-size:0.9rem;font-weight:600;">GitHub</a>
    <a class="hero-btn" href="/notes" style="border:1px solid var(--lightgray);color:var(--dark);border-radius:8px;padding:7px 18px;text-decoration:none;font-size:0.9rem;font-weight:600;">浏览笔记</a>
    <a class="hero-btn" href="/index.xml" style="border:1px solid var(--lightgray);color:var(--dark);border-radius:8px;padding:7px 18px;text-decoration:none;font-size:0.9rem;font-weight:600;">RSS 订阅</a>
  </div>
</div>

## 关于我

我是 Kimio，计算机专业本科在读。目前的主线是**强化学习**和**大语言模型**：一边打基础——从 Transformer 到 PPO、GRPO；一边读论文——推理增强、多模态检索（CIR）、算法博弈论都在涉猎。

我习惯把学会的东西整理成笔记公开在这里，因为**写下来，是为了想清楚**。如果你对这些方向感兴趣，欢迎交流。

## 我的笔记

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:0.5rem 0 0.5rem;">
  <a class="home-card" href="/notes/basics" style="display:block;padding:16px 18px;border:1px solid var(--lightgray);border-radius:12px;text-decoration:none;">
    <span style="font-size:1.5rem;">📚</span>
    <div style="font-weight:700;color:var(--dark);margin:6px 0 4px;">基础笔记</div>
    <div style="color:var(--darkgray);font-size:0.88rem;line-height:1.55;">RL 与 LLM 的核心概念——Transformer、PPO、GRPO、InfoNCE……从零开始的知识地基。</div>
  </a>
  <a class="home-card" href="/notes/papers" style="display:block;padding:16px 18px;border:1px solid var(--lightgray);border-radius:12px;text-decoration:none;">
    <span style="font-size:1.5rem;">🔍</span>
    <div style="font-weight:700;color:var(--dark);margin:6px 0 4px;">论文精读</div>
    <div style="color:var(--darkgray);font-size:0.88rem;line-height:1.55;">一篇一篇读：DeepSeek-V3、思维链系列、多模态检索、算法博弈论……</div>
  </a>
  <a class="home-card" href="/notes" style="display:block;padding:16px 18px;border:1px solid var(--lightgray);border-radius:12px;text-decoration:none;">
    <span style="font-size:1.5rem;">🗂️</span>
    <div style="font-weight:700;color:var(--dark);margin:6px 0 4px;">全部笔记</div>
    <div style="color:var(--darkgray);font-size:0.88rem;line-height:1.55;">浏览完整目录，或直接用左侧搜索找想看的内容。</div>
  </a>
</div>

**从这几篇开始：**

- [Transformer](notes/basics/Transformer)：注意力机制与整体架构，一切的起点
- [PPO](notes/basics/PPO) · [GRPO](notes/basics/GRPO) · [GAE](notes/basics/GAE)：策略优化一条线
- [DeepSeek-V3](notes/papers/DeepSeek-V3)：MLA、MoE 与训练目标的改造之路
- [CoT](notes/papers/CoT-Paper-Report) → [Self-Consistency](notes/papers/Self-Consistency-Paper-Report) → [Tree-of-Thoughts](notes/papers/Tree-of-Thoughts-Paper-Report)：推理增强三部曲
- [Pacing Equilibria](notes/papers/Pacing-Equilibria)：预算约束拍卖中的平滑投放均衡（算法博弈论）

## 找到我

- GitHub：[@kimio-hub](https://github.com/kimio-hub)——我的代码和项目都在这里
- 订阅更新：<a href="/index.xml">RSS</a>——第一时间收到新笔记

本站由 [Quartz](https://quartz.jzhao.xyz) 构建，托管在 Cloudflare。笔记先写在本地 Obsidian，筛选后一键同步发布。

> 写下来，是为了想清楚。
